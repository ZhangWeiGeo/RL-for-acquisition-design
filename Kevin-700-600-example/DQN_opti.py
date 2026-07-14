'''
    This code is used for deep-Q leanring to optimize the source/receiver spatial positions
    
    see:
        Algorithm 1:   Human-level control through deep reinforcement learning, 2015
        Algorithm 8.3: Deep Q-learning (off-policy version), page 184 
        Zhao, S., 2025, Mathematical foundations of reinforcement learning: Springer Nature Press and Tsinghua University Press.
'''

import sys
import os
home_path = os.getenv("HOME"); 
sys.path.append("/home/zhangjiwei/pyfunc/lib");

from lib_sys import *
device, cp_device  = WR.get_device_ini(0)
# WR.get_seed()

from lib_pylops import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os


# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# -------------------------------
# 1. SourcePlacement Environment
# -------------------------------
class SourcePlacementEnv:
    def __init__(self, total_num=9, shot_num=3):
        """
            - total_num: 总可选格子数（索引从0到total_num-1）
            - shot_num: 需要放置的震源数目
            状态：长度total_num的0/1数组，1表示此格子有震源
            动作：每个震源都可左移/右移（约束条件内）
        """
        self.total_num      = total_num
        self.shot_num       = shot_num
        self.interval       = total_num//shot_num
        self.state_dim      = total_num
        self.action_dim     = shot_num * 2  # left/right for each source index
        self.state = None
        self.reset()

    def reset(self, random_type=0):
        """
            初始化状态：放shot_num个震源到total_num个格子。
            random_type=0: 全随机；random_type=1: jetter采样，分区
        """
        self.state = np.zeros(self.total_num, dtype=np.float32)
        
        if random_type==0:
            indices = np.random.choice(self.total_num, self.shot_num, replace=False)
        else:
            indices = []
            for ishot in range(self.shot_num):
                start = ishot*self.interval
                end   = min((ishot+1)*self.interval, self.total_num)
                idx = np.random.randint(start, end)
                indices.append(idx)
        
        self.state[indices] = 1.0
        return self.state.copy()

    def valid_actions(self, state, random_type=0):
        """
            返回可行动作列表：每个动作为(物理位置, 方向)
            物理位置是state里的索引，方向-1左/+1右
            同时返回每个动作对应DQN输出索引
        """
        action_valid   = []
        action_indices = []
        positions = np.where(state == 1)[0]
        for ishot, pos in enumerate(positions):
            
            if random_type==0:
                start = 0
                end   = self.total_num - 1
            else:
                start = ishot*self.interval
                end   = min((ishot+1)*self.interval, self.total_num)
            
            '''left'''
            if pos > start and state[pos - 1] == 0:
                action_valid.append( (pos, -1) )
                action_indices.append( ishot*2 )
            '''right'''
            if pos < end - 1 and state[pos + 1] == 0:
                action_valid.append( (pos, +1) )
                action_indices.append( ishot*2+1 )
        
        return action_valid, action_indices
    
    def step(self, action, random_type=0, 
                                 
                             reward_way  = 1, 
                             hessian_list= [], 
                             W_matrix    = None, 
                             normalized_hes=True,
                             evaluate_bool = True,
                             
                             var_axis    = (0,1),
                             ):
        """
            执行动作: (physical_pos, direction)，
            返回新状态与奖励
        """
        pos, direction = action;
        new_pos = max(0, min(pos + direction, self.total_num - 1));
        s2 = self.state.copy();
        s2[pos] = 0.0;
        s2[new_pos] = 1.0;
        if evaluate_bool:
            reward = self.evaluate(self.state, s2, 
                                   
                                   reward_way       = reward_way, 
                                   hessian_list     = hessian_list, 
                                   W_matrix         = W_matrix, 
                                   normalized_hes   = normalized_hes,
                                   
                                   
                                   var_axis         = var_axis,
                                   );
        else:
            reward = 0.0;
        
        valid_actions, _ = self.valid_actions(s2, random_type=random_type);
        done = (len(valid_actions) == 0);
        
        self.state = s2.copy();
        return s2, reward, done

    def evaluate(self, s1, s2, 
                                reward_way  = 1, 
                                hessian_list= [], 
                                W_matrix    = None, 
                                normalized_hes = True,
                                
                                var_axis    = (0,1),
                ):
        """
                reward_way==0   最大所有震源间隔的和
                
                reward_way==1:  最小化震源间隔的方差
        """
        pos1   = np.sort(np.where(s1 == 1)[0]);
        pos2   = np.sort(np.where(s2 == 1)[0]);
        reward = 0;
        
        if len(pos2) > 1:
            
            diffs1   = np.diff(pos1);
            diffs2   = np.diff(pos2);
            sum_gap1 = np.sum(diffs1);
            sum_gap2 = np.sum(diffs2);
            
            if   reward_way==0:
                '''Maximize the spacing of the source'''
                reward = sum_gap2;
            
            elif reward_way==1:
                reward = -np.var(diffs2);
                
            elif reward_way==2:
                '''
                    compute <m,  Hm>,    m is W_matrix
                '''
                hes_gridz = hessian_list[-3]
                hes_gridx = hessian_list[-2]
                hes_ref   = hessian_list[-1]
                
                psf_list  = hessian_list[0:-3]
                
                subset_list1 = [ psf_list[i] for i in pos1 ];
                subset_list2 = [ psf_list[i] for i in pos2 ];
                
                psfs1     = WR.list_arr_sum_element( subset_list1 ,  module=cp);
                psfs2     = WR.list_arr_sum_element( subset_list2 ,  module=cp);
                inv_dims  = list(hes_ref.shape);
            
                if normalized_hes:
                    psfs1 = WR.arr_normalized(psfs1, axis=(2,3), abs_bool=True);
                    psfs2 = WR.arr_normalized(psfs2, axis=(2,3), abs_bool=True);
                
                
                '''inner dot 1'''
                cp_op1 = NonStationaryConvolve2D(dims=inv_dims, hs=psfs1, ihx=hes_gridz, ihz=hes_gridx, dtype=cp.float32, engine="cuda");
                
                sys1   = cp_op1@hes_ref
                inner_dot_value1 = cp.sum(  sys1 * hes_ref  ).item();
                
                '''inner dot 2'''
                cp_op1 = NonStationaryConvolve2D(dims=inv_dims, hs=psfs2, ihx=hes_gridz, ihz=hes_gridx, dtype=cp.float32, engine="cuda");
                
                sys2   = cp_op1@hes_ref
                inner_dot_value2 = cp.sum(  sys2 * hes_ref  ).item();
                
                if inner_dot_value2 > inner_dot_value1:
                    reward = 10;
                else:
                    reward = -10;
                
                # reward = 1000 * (inner_dot_value2 - inner_dot_value1);
                
                # print(f"inner_dot_value1={inner_dot_value1}, inner_dot_value2={inner_dot_value2}, reward={reward} \n");
                
                # PF.imshow(sys2, output_name="sys2.png", vmin_scale=0.5, vmax_scale=0.5);
            
            
            elif reward_way==3:
                '''
                    compute <m,  Hm>,    m is W_matrix
                '''
                hes_gridz = hessian_list[-3]
                hes_gridx = hessian_list[-2]
                hes_ref   = hessian_list[-1]
                
                psf_list  = hessian_list[0:-3]
                
                subset_list1 = [ psf_list[i] for i in pos1 ];
                subset_list2 = [ psf_list[i] for i in pos2 ];
                
                psfs1     = WR.list_arr_sum_element( subset_list1 ,  module=cp);
                psfs2     = WR.list_arr_sum_element( subset_list2 ,  module=cp);
                inv_dims  = list(hes_ref.shape);
                
                if normalized_hes:
                    psfs1 = WR.arr_normalized(psfs1, axis=(2,3), abs_bool=True);
                    psfs2 = WR.arr_normalized(psfs2, axis=(2,3), abs_bool=True);
                
                
                '''inner dot 1'''
                cp_op1 = NonStationaryConvolve2D(dims=inv_dims, hs=psfs1, ihx=hes_gridz, ihz=hes_gridx, dtype=cp.float32, engine="cuda");
                
                sys1   = cp_op1@hes_ref
                inner_dot_value1 = cp.sum(  sys1 * hes_ref  ).item();
                
                '''inner dot 2'''
                cp_op1 = NonStationaryConvolve2D(dims=inv_dims, hs=psfs2, ihx=hes_gridz, ihz=hes_gridx, dtype=cp.float32, engine="cuda");
                
                sys2   = cp_op1@hes_ref
                inner_dot_value2 = cp.sum(  sys2 * hes_ref  ).item();
                
                # if inner_dot_value2 > inner_dot_value1:
                #     reward = 10;
                # else:
                #     reward = -10;
                
                reward = 100 * (inner_dot_value2 - inner_dot_value1);
                
                # print(f"inner_dot_value1={inner_dot_value1}, inner_dot_value2={inner_dot_value2}, reward={reward} \n");
                
                # PF.imshow(sys2, output_name="sys2.png", vmin_scale=0.5, vmax_scale=0.5);
                
                
            
            elif reward_way==4:
                '''optimize the illumination strength'''
                total_psf_num = np.prod( list(hessian_list[0].shape)[0:2] )
                
                subset_list1 = [ hessian_list[i] for i in pos1 ];
                subset_list2 = [ hessian_list[i] for i in pos2 ];
                
                hessian1     = WR.list_arr_sum_element( subset_list1 ,  module=cp);
                hessian2     = WR.list_arr_sum_element( subset_list2 ,  module=cp);
                
                illumination_1 = np.sum( hessian1,   axis=(2,3) );
                illumination_2 = np.sum( hessian2,   axis=(2,3) );
                
                improvement_mask = (illumination_2 > illumination_1)    
                deteriorate_mask = (illumination_2 < illumination_1)    
                
                reward = (np.sum(improvement_mask) - np.sum(deteriorate_mask)) / total_psf_num;
                
                # print(f"illumination_2: {illumination_2.shape}")
                # print(f"reward: {reward}")
            
            elif reward_way==5:
                
                total_psf_num = np.prod( list(hessian_list[0].shape)[0:2] )
                
                subset_list1 = [ hessian_list[i] for i in pos1 ];
                subset_list2 = [ hessian_list[i] for i in pos2 ];
                ''' hessian1 '''
                focus_index1, focus_var1 = WR.acqustion_hessian_evaluate_focus( subset_list1, W_matrix, var_axis=var_axis );
                
                ''' hessian2 '''
                focus_index2, focus_var2 = WR.acqustion_hessian_evaluate_focus( subset_list2, W_matrix, var_axis=var_axis );
                
                
                # --- 先计算前后聚集度指标 ---
                # 1. 逐位置比较聚集度变化
                improvement_mask = (focus_index2 < focus_index1)    # better True
                deteriorate_mask = (focus_index2 > focus_index1)    # worse True
                
                reward_psf = np.sum(improvement_mask) - np.sum(deteriorate_mask)
                
                
                # 2. compare var
                # if focus_var2 < focus_var1:
                #     reward_var = total_psf_num;
                # else:
                #     reward_var = -total_psf_num;
                reward_var = 0.0;
                
                # 3. reward,   normalized reward by total_psf_num
                reward = (reward_psf + reward_var) / total_psf_num;
                
                # --- 输出判分情况（便于调试和后续分析）---
                # print(f"PSF聚焦变好数量: {np.sum(improvement_mask)}")
                # print(f"PSF聚焦变差数量: {np.sum(deteriorate_mask)}")
                # print(f"聚焦均匀性 var1={focus_var1:.3f} -> var2={focus_var2:.3f}, var改进分={reward_var}")
                # print(f"step reward: {reward}")
        
        return reward

# -------------------------------
# 2. DQN network
# -------------------------------
class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.indims = state_dim
        self.oudims = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------
# 3. Experience replay buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32),      # s
            torch.tensor(a, dtype=torch.int64),                  # a
            torch.tensor(r, dtype=torch.float32),                # r
            torch.tensor(np.stack(s2), dtype=torch.float32),     # s2
            torch.tensor(done, dtype=torch.float32)              # done (0. or 1.)
        )

    def __len__(self):
        return len(self.buffer)