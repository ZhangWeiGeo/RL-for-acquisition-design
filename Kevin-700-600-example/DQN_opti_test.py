


from DQN_opti import *

sys.path.append("/home/zhangjiwei/pyfunc/acq_info/Kevin-700-600-modify/")
from hess_info_compute     import *

from main_para     import *


hes_gridz            = hes_para_obj.gridz - hes_para_obj.gridz[0]
hes_gridx            = hes_para_obj.gridx - hes_para_obj.gridx[0]
target_hes_ref       = angle_ref_3D[0,...][hes_para_obj.start_z:hes_para_obj.nz, hes_para_obj.start_x:hes_para_obj.nx]
inv_dims  = list(target_hes_ref.shape);


'''
    generate the slope reflectivity for target imaging
'''
slope_hes_ref_list   = []

# for height in range(0, 1):
#     tmp_ref        = WR.generate_slope_line(shape_list=inv_dims, angle_degree=135,
#                                                   value=1.0, 
#                                                   middle_point=[inv_dims[0]//2, inv_dims[1]//2 - height]);
#     slope_hes_ref_list.append( tmp_ref );

# slope_hes_ref = WR.list_arr_sum_element(  slope_hes_ref_list  );
# slope_hes_ref[0:70, 0:70] = 0.0;

slope_vp_load   = np.load(path_dict['vel'] + "slope_vp.npz")
slope_vp        = slope_vp_load['arr_0']
slope_hes_ref   = vector_ref_dz_arr[hes_para_obj.slice_2D]
slope_hes_vp    = slope_vp[hes_para_obj.slice_2D];
output1         = WR.array_fd_op_first_derivative_stagger(slope_hes_vp, 
                                       axis=0, 
                                       sampling=1.0,
                                       forward=True)

output2 = np.where(output1 != 0, slope_hes_ref, output1)
slope_hes_ref[...]   = 1.0 * output2[...]

PF.imshow(  target_hes_ref, output_name="target_hes_ref.png", vmin_scale=0.5, vmax_scale=0.5, cmap="seismic", x1beg=hes_para_obj.plot_x_beg*0.001, x2beg=hes_para_obj.plot_z_beg*0.001);

PF.imshow(  slope_hes_ref, output_name="slope_hes_ref.png", vmin_scale=0.5, vmax_scale=0.5, cmap="seismic", x1beg=hes_para_obj.plot_x_beg*0.001, x2beg=hes_para_obj.plot_z_beg*0.001);

PF.imshow(  slope_hes_vp, output_name="slope_hes_vp.png", vmin_scale=0.5, vmax_scale=0.5, cmap="seismic", x1beg=hes_para_obj.plot_x_beg*0.001, x2beg=hes_para_obj.plot_z_beg*0.001);

PF.imshow(  output1, output_name="output1.png", vmin_scale=0.5, vmax_scale=0.5, cmap="seismic", x1beg=hes_para_obj.plot_x_beg*0.001, x2beg=hes_para_obj.plot_z_beg*0.001);
PF.imshow(  output2, output_name="output2.png", vmin_scale=0.5, vmax_scale=0.5, cmap="seismic", x1beg=hes_para_obj.plot_x_beg*0.001, x2beg=hes_para_obj.plot_z_beg*0.001);




# -------------------------------
# 4.other parameters for reward
# -------------------------------'
hessian_file      = f"/home/zhangjiwei/pyfunc/acq_info/Kevin-700-600-modify/ray/ray-data/acqu_info_full70-shot_num_dependent_hessian_dict.npz";
hessian_dict      = WR.dict_read(  hessian_file  );

ini_hessian_list  = WR.dict_value_to_list(hessian_dict);
ini_hessian_list  = [ np.squeeze( array )  for array in   ini_hessian_list];
# ini_hessian_list      = [ WR.array_squeeze( array )  for array in   ini_hessian_list];
ini_psf_wz, ini_psf_wx    = list(ini_hessian_list[0].shape)[2:4];


# normalized_bool   = False;
psf_wz, psf_wx    = ini_psf_wz, ini_psf_wx;

hessian_slice     = (slice(None), slice(None), slice(ini_psf_wz//2-psf_wz//2, ini_psf_wz//2+psf_wz//2+1), slice(ini_psf_wx//2-psf_wx//2, ini_psf_wx//2+psf_wx//2+1) );

ini_hessian_list  = [ hessian[hessian_slice] for hessian in ini_hessian_list ];


W_matrix          = SF.make_linear_weight_matrix(shape=[psf_wz, psf_wx], gamma=1);
W_matrix          = W_matrix[None, None, :, :];


if normalized_bool:
    hessian_list = [];
    '''  normalized the hessian '''
    for idx, hessian1 in enumerate(  ini_hessian_list  ):
        hessian1    = hessian1 / np.max( np.abs(hessian1), axis=(2, 3), keepdims=True );
        hessian_list.append(  hessian1.copy()  );
else:
    hessian_list    = ini_hessian_list;


'''
    load to cupy
'''
hessian_list = WR.list_to_cupy(  hessian_list  );


total_num    = len(hessian_list);
# shot_num     = 31;
shot_interval= total_num//shot_num

if total_num<=shot_num:
    sys.exit(f"total_num<=shot_num: {total_num}<={shot_num}");


'''   '''
output_path = f"./output-nor-{normalized_bool}-shot_num-{shot_num}-random-{random_type}/"; 
output_path = WR.get_unique_dir(output_path);  WR.mkdir(output_path);


log_file    = output_path + "log.txt";
WR.write_txt(log_file, "", w_type="w+");
try:
    for optimize_source in optimize_source_list:
        write_txt(log_file, f"optimize_source={optimize_source}\n", w_type="a+");
except:
    ''''''



'''  tmp plot psf for check'''
hessian_part    = WR.list_arr_sum_element( hessian_list[0::shot_interval], module=cp );
hessianfull     = WR.list_arr_sum_element( hessian_list, module=cp );

for iz in range(     hessianfull.shape[0]//2, hessianfull.shape[0]//2+2):
    for ix in range(     hessianfull.shape[1]//2, hessianfull.shape[1]//2+2):
        
        try:
            patch    = hessian_part[iz, ix, ...];
            binname  = WR.bin_name(patch);
            filename = output_path + f"part-{binname}-z-{iz}-x-{ix}";
            
            PF.imshow( patch, output_name=f"{filename}.png"  );
            WR.write_file(f"{filename}.bin", patch);
            
            
            patch    = hessianfull[iz, ix, ...];
            binname  = WR.bin_name(patch);
            filename = output_path + f"full-{binname}-z-{iz}-x-{ix}";
            PF.imshow( patch, output_name=f"{filename}.png"  );
            WR.write_file(f"{filename}.bin", patch);
        except:
            '''  '''


from lib_pylops import *



target_hes_ref   = cp.asarray(target_hes_ref);
slope_hes_ref    = cp.asarray( slope_hes_ref);



inv_dims  = list(target_hes_ref.shape);
cp_op1    = NonStationaryConvolve2D(dims=inv_dims, hs=hessian_part, 
                                    ihx=hes_gridz, ihz=hes_gridx, 
                                    dtype=cp.float32, engine="cuda");


'''
    there is important to normalized the PSF, 
    it means that we need to compensate for the uneven illumination effects
'''
target_sys1      = cp_op1@target_hes_ref;
PF.imshow(target_sys1, output_name=output_path+"target_sys1.png", vmin_scale=0.5, vmax_scale=0.5);
slope_sys1       = cp_op1@slope_hes_ref;
PF.imshow(slope_sys1, output_name=output_path+"slope_sys1.png", vmin_scale=0.5, vmax_scale=0.5);





hessian_list = hessian_list + [hes_gridz] + [hes_gridx]  + [cp.asarray(slope_hes_ref)];





# -------------------------------
# 5.train,  find optimal solution
# -------------------------------
# num_episodes        = 800;

# num_each_episodes   = 5*num_episodes;
# max_predict_steps         = 50*num_episodes;
# update_target_loop  = total_num//2;
# log_every_episode   = 50;

# batch_size        = 128
# buffer_capacity   = 10*total_num;
gamma             = 0.99;
lr                = 1e-4;

'''
    In the early stage of the iteration, a larger exploration can be adopted. In the later stage, it is set to 0 and no random exploration is conducted
'''
# epsilon_start   = 0.5;
# epsilon_end     = 0;
epsilon_arr     = np.linspace(epsilon_start, epsilon_end,  num_episodes);
                
dqn_convergence = [100, 0.00001];
                
                
env             = SourcePlacementEnv(total_num=total_num, shot_num=shot_num);
state_dim       = env.state_dim;
action_dim      = env.action_dim;
                
# reward_way      = 2;
normalized_hes  = False;
# random_type     = 0;


'''
    log information
'''
local_vars = locals();
sys_info_dict         = {};
for key, value in list(local_vars.items()): 
    if isinstance(value, (int, float, bool, str)):
        if key not in ["sys_info_dict", "parameters_arr"]:  
            sys_info_dict[key] = f"{value} (type: {type(value).__name__})";
            
for key, value in sys_info_dict.items():
    print(f"{key}: {value}\n");
WR.dict_write_as_txt(log_file, sys_info_dict, w_type='a+');



'''
    allocate network and share the parameter
'''
policy_net      = DQNNet(state_dim, action_dim).to(device);
target_net      = DQNNet(state_dim, action_dim).to(device);
target_net.load_state_dict( policy_net.state_dict() );

optimizer       = optim.Adam(policy_net.parameters(), lr=lr);
replay_buffer   = ReplayBuffer(buffer_capacity);


rewards_history     = [];
loss_history        = [];

file_dict   = {}


'''
    train: DQN
'''
if train_bool:
    
    for episode in range( num_episodes ):
        
        epsilon = epsilon_arr[episode];
        
        '''
            generate a random current_state
        '''
        current_state   = env.reset(random_type=random_type);
        episode_reward  = 0;
        episode_loss    = 0;
        
        
        for inner_loop in range( num_each_episodes ):
            
            '''
                generate the valid_actions space and valid_action_indices
                \eplison behavior policy
            '''
            valid_actions, valid_action_indices = env.valid_actions(current_state, random_type=random_type);
            
            if len(valid_actions) == 0:
                break
            
            if np.random.rand() < epsilon:
                idx         = random.randint(0, len(valid_actions) - 1);
                action      = valid_actions[idx];
                action_idx  = valid_action_indices[idx];
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0);
                    '''the function of actions f(a)'''
                    q_values     = policy_net(state_tensor).cpu().numpy()[0];
                    '''argmax( f ),   valid_actions is a small set of total'''
                    q_valid      = np.array([q_values[i] for i in valid_action_indices]);
                    best_idx     = np.argmax( q_valid );
                    
                    action       = valid_actions[best_idx];
                    action_idx   = valid_action_indices[best_idx];
            
            '''
                Store the experience samples generated by \pi_b
                ( s, a, r, s' )
                ( current_state, action_idx, reward, next_state )
            '''
            next_state, reward, done = env.step( action, 
                                                random_type =random_type,
                                                reward_way  =reward_way, 
                                                hessian_list=hessian_list, 
                                                W_matrix    =W_matrix,
                                                normalized_hes=normalized_hes,
                                                evaluate_bool = True,
                                                );
            
            episode_reward          += reward;
            
            
            
            replay_buffer.push(current_state, action_idx, reward, next_state, done);
            current_state   = next_state;
            
            
            '''
                when len(replay_buffer) >= batch_size
                optimize the network
            '''
            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, s1_batch, done_batch = replay_buffer.sample(batch_size);
                s_batch     = s_batch.to(device);
                a_batch     = a_batch.to(device);
                r_batch     = r_batch.to(device);
                s1_batch    = s1_batch.to(device);
                done_batch  = done_batch.to(device);
                
                '''Y_T = r_batch + gamma * max(q_next)   2025'''
                '''Y_j = r_j + gamma *  max_a' q(s_j+1, a')       2015'''
                with torch.no_grad():
                    q_next   = target_net(s1_batch).max(1)[0];
                    
                    q_target = r_batch + gamma * q_next * (1 - done_batch);
                
                '''
                    compute Q(s_j, a) using network
                    get the action value at a_batch, you specfiy the actions (a_batch)
                '''
                q_values = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1);
                
                loss = nn.MSELoss()(q_values, q_target);
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                
                episode_loss+=loss.item();
                
                # print(f"a_batch={a_batch}");
                # print(f"policy_net(s_batch).shape={policy_net(s_batch).shape}");
                # print(f"a_batch.shape={a_batch.shape}");
                # print(f"a_batch.unsqueeze(1).shape={a_batch.unsqueeze(1).shape}");
                # print(f"q_target.shape={q_target.shape}");
                
                
                
            if inner_loop>= batch_size and inner_loop % update_target_loop == 0:
                target_net.load_state_dict( policy_net.state_dict() );
        
        
        rewards_history.append( episode_reward );
        loss_history.append(    episode_loss   );
        
        
        
        '''
            at each episode to output log
        '''
        state_pos  = np.sort( np.where(next_state == 1 )[0]);
        state_diff       = np.diff( state_pos).astype( np.int32 );
        state_var        = np.var(  state_diff );
        
        
        file_dict['1'] = f"Episode {episode+1}";
        file_dict['2'] = f"state_pos      = {state_pos}";
        file_dict['3'] = f"state_diff     = {state_diff}";
        file_dict['4'] = f"state_var      = {state_var:.3f}";
        file_dict['5'] = f"episode_reward = {episode_reward:.2f}";
        file_dict['6'] = f"loss           = {episode_loss:.2f}";
        file_dict['7'] = f"epsilon        = {epsilon:.2f}\n";
        
        for key, value in file_dict.items():
            write_txt(  log_file, value, print_bool=True );
            
        if episode % log_every_episode == 0:
            
            optimize_source_arr = np.array( state_pos , dtype=np.int32 )
            np.savez(output_path + f"optimize_source_arr-{episode}.npz", optimize_source_arr)
            
            torch.save(policy_net.state_dict(), output_path + f'policy-net-{episode}.pth');
            
            
            # ----- rewards_history -----
            plt.figure(figsize=(8, 5), dpi=200)  #
            plt.plot(rewards_history, linewidth=2)
            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Total reward per Episode', fontsize=14)
            plt.title('DQN Source Placement: Episode Reward', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path+f"{state_dim}-{action_dim}-rewards_history.png")
            plt.close()
            
            # ----- loss_history -----
            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(loss_history, linewidth=2)
            plt.xlabel('Episode', fontsize=14)
            plt.ylabel('Q-Value Loss', fontsize=14)
            plt.title('DQN Source Placement: Q-Value Loss', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path+f"{state_dim}-{action_dim}-loss_history.png")
            plt.close()
            # reward收敛才是强化学习里最关键的收敛标准！ 
            # deep_learning loss是否微涨，只要reward稳定，不用过于担心
        
        # if episode > dqn_convergence[0]:
        #     avg_reward_prev = np.mean(rewards_history[-dqn_convergence[0]:-1]);
        #     if avg_reward_prev != 0:
        #         rel_change = abs(episode_reward - avg_reward_prev) / abs(avg_reward_prev);
        #         if rel_change < dqn_convergence[1]:
                    
        #             file1 = f"\n Early stop at episode {episode+1}! Relative reward change < {dqn_convergence[1]}%";
        #             file2 = f"best_state = {best_state.astype(int)}";
        #             file3 = f"best_diff  = {best_diff.astype(int)}";
                    
        #             write_txt(  log_file, file1 + file2 + file3, print_bool=True );
                    
        #             break;


if train_bool:
    state_pos           = np.sort( np.where(next_state == 1 )[0]);
    optimize_source_arr = np.array( state_pos , dtype=np.int32 )
    np.savez(output_path + "optimize_source_arr_final1.npz", optimize_source_arr);
    
    final_hessian_list = [];
    for idx in optimize_source_arr:
        final_hessian_list.append(   hessian_list[idx]    );
    
    hessian_final    = WR.list_arr_sum_element( final_hessian_list, module=np );
    
    cp_op1    = NonStationaryConvolve2D(dims=inv_dims, hs=hessian_final, 
                                        ihx=hes_gridz, ihz=hes_gridx, 
                                        dtype=cp.float32, engine="cuda");
    
    target_sys_final = cp_op1@target_hes_ref;
    slope_sys_final  = cp_op1@slope_hes_ref;
    
    
    PF.imshow(target_sys_final, output_name=output_path + "target_sys_final1.png", 
              vmin_scale=0.5, vmax_scale=0.5);
    PF.imshow(slope_sys_final,  output_name=output_path + "slope_sys_final1.png",  
              vmin_scale=0.5, vmax_scale=0.5);
    
    
    inner_dot_value_final1 = cp.sum(  target_sys_final * target_hes_ref  ).item();
    inner_dot_value_final2 = cp.sum(  slope_sys_final  * slope_hes_ref   ).item();
    
    
    write_txt(  log_file, f"dot1={inner_dot_value_final1}", print_bool=True );
    write_txt(  log_file, f"dot2={inner_dot_value_final2}", print_bool=True );







'''
    predict:
'''
write_txt(  log_file, f" DQN Train is done \n DQN Train is done \n ", print_bool=True );


policy_net_file = "/home/zhangjiwei/pyfunc_test/test/Kevin-700-600-modify/A-RL/main5-plot/run-60-plot/policy-net-final.pth"


if predict_bool:
    
    if len(policy_net_file)!=0:
        policy_net.load_state_dict(torch.load(policy_net_file))
        policy_net.eval()
        write_txt(  log_file, f"policy_net_file={policy_net_file}", print_bool=True );
    
    
    current_state   = env.reset(random_type=random_type);
    
    for inner_loop in range(0, max_predict_steps):
        
        valid_actions, valid_action_indices = env.valid_actions(current_state, random_type=random_type);
        
        if len(valid_actions) == 0:
            break
        
        with torch.no_grad():
            state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0);
            '''the function of actions f(a)'''
            q_values     = policy_net(state_tensor).cpu().numpy()[0];
            '''argmax( f ),   valid_actions is a small set of total'''
            q_valid      = np.array([q_values[i] for i in valid_action_indices]);
            best_idx     = np.argmax( q_valid );
            
            action       = valid_actions[best_idx];
            action_idx   = valid_action_indices[best_idx];
            
        next_state, reward, done = env.step( action, 
                                            random_type =random_type,
                                            reward_way  =reward_way, 
                                            hessian_list=hessian_list, 
                                            W_matrix    =W_matrix,
                                            normalized_hes=normalized_hes,
                                            evaluate_bool = False,
                                            );
        
        
        if inner_loop % log_every_episode == 0:
            curr_state_pos      = np.sort( np.where(current_state == 1 )[0]);
            next_state_pos      = np.sort( np.where(next_state == 1 )[0]);
            write_txt(  log_file, f"predict={inner_loop}, curr_state_pos={curr_state_pos}", print_bool=True );
            write_txt(  log_file, f"predict={inner_loop}, next_state_pos={next_state_pos}", print_bool=True );
            write_txt(  log_file, f"predict={inner_loop}, action={action}", print_bool=True );
            write_txt(  log_file, f"predict={inner_loop}, action_idx={action_idx}", print_bool=True );
            
            if np.array_equal(curr_state_pos, next_state_pos):
                write_txt(log_file, f"Converged at step {inner_loop}: curr_state_pos == next_state_pos", print_bool=True);
        
        current_state = next_state;



'''
    save    .pth
'''
torch.save(policy_net.state_dict(), output_path + f'policy-net-final.pth');


'''
    output  .npz
'''
if predict_bool:
    
    
    state_pos      = np.sort( np.where(next_state == 1 )[0]);
    optimize_source_arr = np.array( state_pos , dtype=np.int32 )
    np.savez(output_path + "optimize_source_arr_final2.npz", optimize_source_arr);



    final_hessian_list = [];
    for idx in optimize_source_arr:
        final_hessian_list.append(   hessian_list[idx]    );
    
    hessian_final    = WR.list_arr_sum_element( final_hessian_list, module=np );
    
    cp_op1    = NonStationaryConvolve2D(dims=inv_dims, hs=hessian_final, 
                                        ihx=hes_gridz, ihz=hes_gridx, 
                                        dtype=cp.float32, engine="cuda");
    
    target_sys_final = cp_op1@target_hes_ref;
    slope_sys_final  = cp_op1@slope_hes_ref;
    
    PF.imshow(target_sys_final, 
              output_name=output_path + "target_sys_final2.png", 
              vmin_scale=0.5, vmax_scale=0.5);

    PF.imshow(slope_sys_final,  
              output_name=output_path + "slope_sys_final2.png",  
              vmin_scale=0.5, vmax_scale=0.5);
    
    
    inner_dot_value_final1 = cp.sum(  target_sys_final * target_hes_ref  ).item();
    inner_dot_value_final2 = cp.sum(  slope_sys_final  * slope_hes_ref   ).item();
    
    
    write_txt(  log_file, f"dot1={inner_dot_value_final1}", print_bool=True );
    write_txt(  log_file, f"dot2={inner_dot_value_final2}", print_bool=True );
    









from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

# =====================================================
# common plot parameters
# =====================================================
figsize = (4, 2.7)

fontsize_label = 10
fontsize_tick  = 9
fontsize_cbar  = 9

plotx_beg = hes_para_obj.start_x * dx * 0.001
plotz_beg = hes_para_obj.start_z * dz * 0.001

test_path = "./final_result1/"
WR.mkdir(test_path)

if True:
    
    
    # =====================================================
    # load optimized source indices
    # =====================================================
    source_load = np.load("optimize_source_arr_final1.npz")
    
    print(source_load.files)
    
    # load array
    optimize_source_arr = source_load["arr_0"]
    
    print(optimize_source_arr)

    # =====================================================
    # source selection
    # =====================================================
    source_arr1 = np.linspace(0, total_num - 1, 70).astype(np.int32)

    # idx_arr = env.reset(random_type=1)
    # source_arr2 = np.where(idx_arr != 0)[0]
    
    source_arr2 = np.sort(
    np.random.choice(
        np.arange(0, total_num),
        size=70,
        replace=False,
    )).astype(np.int32)
    

    # =====================================================
    # hessian from optimized sources
    # =====================================================
    final_hessian_list = []
    for idx in optimize_source_arr:
        final_hessian_list.append(hessian_list[idx])

    hessian1 = WR.list_arr_sum_element(final_hessian_list)

    # =====================================================
    # hessian from uniformly selected sources
    # =====================================================
    final_hessian_list = []
    for idx in source_arr1:
        final_hessian_list.append(hessian_list[idx])

    hessian2 = WR.list_arr_sum_element(final_hessian_list)

    # =====================================================
    # hessian from randomly selected sources
    # =====================================================
    final_hessian_list = []
    for idx in source_arr2:
        final_hessian_list.append(hessian_list[idx])

    hessian3 = WR.list_arr_sum_element(final_hessian_list)

    # =====================================================
    # apply Hessian operators
    # =====================================================
    cp_op1 = NonStationaryConvolve2D(
        dims=inv_dims,
        hs=hessian1,
        ihx=hes_gridz,
        ihz=hes_gridx,
        dtype=cp.float32,
        engine="cuda",
    )
    sys_final1 = cp_op1 @ target_hes_ref

    cp_op2 = NonStationaryConvolve2D(
        dims=inv_dims,
        hs=hessian2,
        ihx=hes_gridz,
        ihz=hes_gridx,
        dtype=cp.float32,
        engine="cuda",
    )
    sys_final2 = cp_op2 @ target_hes_ref

    cp_op3 = NonStationaryConvolve2D(
        dims=inv_dims,
        hs=hessian3,
        ihx=hes_gridz,
        ihz=hes_gridx,
        dtype=cp.float32,
        engine="cuda",
    )
    sys_final3 = cp_op3 @ target_hes_ref





    # =====================================================
    # plot source distribution
    # =====================================================
    source_x_optimized = optimize_source_arr / (total_num - 1) * 7.0
    source_x_uniform   = source_arr2 / (total_num - 1) * 7.0
    
    plt.figure(figsize=(6, 1.8))
    
    plt.scatter(
        source_x_uniform,
        np.ones_like(source_x_uniform),
        marker="*",
        s=18,
        color="tab:blue",
        alpha=0.85,
    )
    
    plt.scatter(
        source_x_optimized,
        2 * np.ones_like(source_x_optimized),
        marker="*",
        s=28,
        color="tab:red",
        alpha=0.85,
    )
    
    plt.xlim(0, 7)
    plt.ylim(0.5, 2.5)
    
    plt.xlabel("Distance (km)", fontsize=12)
    
    plt.yticks(
        [1, 2],
        ["Random", "Optimized"],
        fontsize=12,
    )
    
    plt.xticks(np.arange(0, 8, 1), fontsize=11)
    
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(test_path + "source_distribution_uniform_vs_optimized.png", dpi=300)
    plt.close()













    # =====================================================
    # mask target_hes_ref by slope_hes_ref
    # =====================================================
    new_arr = cp.zeros_like(target_hes_ref)

    mask = (target_hes_ref != 0) & (slope_hes_ref != 0)
    new_arr[mask] = target_hes_ref[mask]

    # =====================================================
    # plot target_hes_ref and sys_final results
    # =====================================================
    plot_arr_list = [
        sys_final1,
        sys_final2,
        sys_final3,
        sys_final1 - sys_final3, 
        target_hes_ref,
        new_arr,
    ]

    plot_name_list = [
        "sys_final1_optimized",
        "sys_final2_uniform",
        "sys_final3_random",
        "sys_final13_res",
        "target_hes_ref",
        "slope_hes_ref",
    ]

    plot_min = None
    plot_max = None

    for plot_idx in range(len(plot_arr_list)):

        plot_arr = plot_arr_list[plot_idx]
        plot_name = plot_name_list[plot_idx]

        if isinstance(plot_arr, cp.ndarray):
            plot_arr = cp.asnumpy(plot_arr)

        plot_arr = np.asarray(plot_arr)

        filename = test_path + f"{plot_name}"

        if plot_idx == 0:
            tmp_max = np.max(np.abs(plot_arr))
            plot_min = -0.15 * tmp_max
            plot_max = +0.15 * tmp_max

        if plot_idx == 4:
            tmp_max = np.max(np.abs(plot_arr))
            plot_min = -0.2 * tmp_max
            plot_max = +0.2 * tmp_max

        extent = [
            plotz_beg,
            plotz_beg + plot_arr.shape[1] * dz * 0.001,
            plotx_beg + plot_arr.shape[0] * dx * 0.001,
            plotx_beg,
        ]

        plt.figure(figsize=figsize)

        plt.imshow(
            plot_arr,
            cmap="gray",
            vmin=plot_min,
            vmax=plot_max,
            extent=extent,
            aspect="auto",
            interpolation="none",
            origin="upper",
        )

        plt.xlabel("Distance (km)", fontsize=fontsize_label, labelpad=6)
        plt.ylabel("Depth (km)", fontsize=fontsize_label)

        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        ax.set_xticks(np.linspace(extent[0], extent[1], 5))
        ax.set_yticks(np.linspace(extent[3], extent[2], 5))
        ax.tick_params(axis="both", labelsize=fontsize_tick)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        cbar = plt.colorbar(fraction=0.046, pad=0.07)

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        
        cbar.formatter = formatter
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=fontsize_cbar)
        
        # 调整 ×10^{-2} 的位置
        cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize_cbar)
        cbar.ax.yaxis.get_offset_text().set_x(2.8)
        cbar.ax.yaxis.get_offset_text().set_y(1.02)

        plt.tight_layout(pad=0.6)
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()

    # =====================================================
    # plot Hessian patch and shifted amplitude spectrum
    # =====================================================
    plot_hessian_list = [hessian1, hessian2, hessian3]
    plot_name_list = ["optimized", "uniform", "random"]

    for iz in range(20, 21):
        for ix in range(8, 9):

            plot_min = None
            plot_max = None

            for plot_idx in range(0, 3):

                plot_hessian = plot_hessian_list[plot_idx]
                plot_name = plot_name_list[plot_idx]

                patch = cp.asnumpy(plot_hessian[iz, ix, ...])

                binname = WR.bin_name(patch)
                filename = test_path + f"{plot_name}-{binname}-z-{iz}-x-{ix}"

                # =====================================================
                # plot Hessian patch
                # =====================================================
                if plot_idx == 0:
                    patch_max = np.max(np.abs(patch))
                    plot_min = -0.4 * patch_max
                    plot_max = +0.4 * patch_max

                extent = [
                    0,
                    patch.shape[1] * dz,
                    patch.shape[0] * dx,
                    0,
                ]

                plt.figure(figsize=figsize)

                plt.imshow(
                    patch,
                    cmap="gray",
                    vmin=plot_min,
                    vmax=plot_max,
                    extent=extent,
                    aspect="auto",
                    interpolation="none",
                    origin="upper",
                )

                plt.xlabel("Distance (m)", fontsize=fontsize_label, labelpad=6)
                plt.ylabel("Depth (m)", fontsize=fontsize_label)

                ax = plt.gca()
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")

                ax.set_xticks(np.linspace(extent[0], extent[1], 5))
                ax.set_yticks(np.linspace(extent[3], extent[2], 5))
                ax.tick_params(axis="both", labelsize=fontsize_tick)
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        
                cbar = plt.colorbar(fraction=0.046, pad=0.07)

                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                
                cbar.formatter = formatter
                cbar.update_ticks()
                cbar.ax.tick_params(labelsize=fontsize_cbar)
                
                # 调整 ×10^{-2} 的位置
                cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize_cbar)
                cbar.ax.yaxis.get_offset_text().set_x(2.8)
                cbar.ax.yaxis.get_offset_text().set_y(1.02)

                plt.tight_layout(pad=0.6)
                plt.savefig(f"{filename}.png", dpi=300)
                plt.close()

                WR.write_file(f"{filename}.bin", patch.T)

                # =====================================================
                # FFT amplitude spectrum
                # =====================================================
                patch_fft = np.fft.fft2(patch)
                amp_spectrum = np.abs(np.fft.fftshift(patch_fft))

                amp_max = np.max(amp_spectrum)
                if amp_max > 0:
                    amp_spectrum = amp_spectrum / amp_max

                nx, nz = patch_fft.shape

                kx_max = np.pi / dx
                kz_max = np.pi / dz

                extent_kxkz = [
                    -kz_max,
                    kz_max,
                    kx_max,
                    -kx_max,
                ]

                plt.figure(figsize=figsize)

                plt.imshow(
                    amp_spectrum,
                    cmap="seismic",
                    vmin=0,
                    vmax=0.5,
                    extent=extent_kxkz,
                    aspect="auto",
                    interpolation="bicubic",
                    origin="upper",
                )

                plt.xlabel(
                    r"Vertical wavenumber (m$^{-1}$)",
                    fontsize=fontsize_label,
                    labelpad=6,
                )
                plt.ylabel(
                    r"Horizontal wavenumber (m$^{-1}$)",
                    fontsize=fontsize_label,
                )

                ax = plt.gca()
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")
                ax.tick_params(axis="both", labelsize=fontsize_tick)

                cbar = plt.colorbar(fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize_cbar)

                plt.tight_layout(pad=0.6)
                plt.savefig(f"{filename}-kxkz.png", dpi=300)
                plt.close()

WR.move_txt_files_to_pwd()


# =====================================================
# PSF position information
# =====================================================
# psf x position = 320 m
# psf z position = 400 m
# start x position = 280 m
# start z position = 320 m
# interval = 5 m
#
# ix = (320 - 280) / 5 = 8
# iz = (400 - 320) / 5 = 20
# =====================================================

