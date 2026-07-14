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



class ReflectivityEnv:
    def __init__(self, lt, envelope_list, forward_op, obs, train_snr=20.0, module=cp):
        """
        lt: 反射率长度
        M: 包络峰值估计数（非零点个数范围为 [M, 2M]）
        forward_op: 可调用的正演算子函数，形式为 forward_op(reflectivity)
        obs: 观测数据（1D array）
        train_snr: 达到该SNR即终止
        module: np 或 cp
        """
        self.lt = lt
        self.envelope_list = envelope_list
        self.forward_op = forward_op
        self.obs = obs
        self.train_snr = train_snr
        self.module = module
        
        
        self.obs_energy  = module.abs(self.obs)**2
        self.prob        = obs / module.sum(obs)

        self.state_dim   = lt;
        self.action_dim  = 2*lt;

    def reset(self, 
                  fixed_state=None,
                  
                  ref_min_max=[-0.1, 0.1], positions_list=[],
                  
              ):
            
        if fixed_state is not None:
            self.state = fixed_state
            
        else:
            if len(positions_list)==0:
                k = self.module.random.randint(self.envelope_list[0], self.envelope_list[1]).item()
                positions = self.module.random.choice(self.lt, k, replace=False, p=self.prob)
            else:
                k         = len( positions_list );
                positions = positions_list;
            
        
            self.state = self.module.zeros(self.lt, dtype=self.module.float32)
            
            if   self.module.__name__ in ['numpy', 'cupy']:
                self.state[positions] = self.module.random.uniform(ref_min_max[0], ref_min_max[1], k)
            elif self.module.__name__ == 'torch':
                self.state[positions] = (ref_min_max[1] - ref_min_max[0]) * self.module.rand(k) + ref_min_max[0]
        
        
        '''shared part'''
        self.simulated = self.forward_op(self.state)
        self.snr1      = self.snr(self.obs, self.simulated);
        
        
        if hasattr(self.state, 'clone'):
            return self.state.clone()
        else:
            return self.state.copy()

    def snr(self, signal, signal2):
        signal_power = self.module.mean(signal ** 2)
        
        noise_power = self.module.mean((signal - signal2) ** 2)
        
        if noise_power == 0:
            return 100
        
        snr = 10 * self.module.log10(signal_power / noise_power)
        
        return snr.item()

    def valid_actions(self, state):
        """
        返回当前state下的可行动作：
        - action_valid: [(idx, +1), (idx, -1)]
        - action_indices: [idx*2, idx*2+1]
        """
        action_valid = []
        action_indices = []
        positions = self.module.where(state != 0.0)[0]
        for idx in positions:
            idx = int(idx)
            action_valid.append((idx, +1))
            action_indices.append(idx * 2)
            action_valid.append((idx, -1))
            action_indices.append(idx * 2 + 1)
        return action_valid, action_indices
    
    def step_and_reward(self, action, current_state, relative_per=0.02):
        
        pos, direction = action;
        if hasattr(current_state, 'clone'):
            s2 = current_state.clone();
        else:
            s2 = current_state.copy();
        
        if direction==1:
            s2[pos]     += relative_per * s2[pos];
        else:
            s2[pos]     -= relative_per * s2[pos];
        
        '''we do not need re-compute'''
        # self.simulated1  = self.forward_op(self.state)
        # self.snr1        = self.snr(self.obs, self.simulated1);
        
        self.simulated2  = self.forward_op(s2);
        self.snr2        = self.snr(self.obs, self.simulated2);
        
        done   = self.snr2 >= self.train_snr
        
        reward = self.snr2 - self.snr1;
        
        '''save the previous snr1 and state'''
        self.snr1        = 1.0 * self.snr2
        
        return s2, reward, done


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