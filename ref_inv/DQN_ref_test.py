


from DQN_ref  import *
home_path = os.getenv("HOME"); 
sys.path.append("/home/zhangjiwei/pyfunc/lib");
from lib_post  import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_path = "./output15/";  os.makedirs(output_path, exist_ok=True);
log_file    = output_path + "log.txt";
WR.write_txt(log_file, "", w_type="w+");

'''prior data'''
module=torch


freq             = 40;
lt               = 200
snr_db           = 10
train_snr        = snr_db;

''''''
wavelet_ricker1  = SF.wavelet_ricker(nt=200, lt=200, dt=0.001, freq=freq, module=cp);
wavelet_ricker2  = SF.phase_rotate_nd_array(wavelet_ricker1, angle_degrees=45, axis=0);
wavelet_ricker1  = WR.array_to_module(wavelet_ricker1, module)
wavelet_ricker2  = WR.array_to_module(wavelet_ricker2, module)


PF.plot_graph([wavelet_ricker1,wavelet_ricker2],
              plot_number =2,
              legend_name =("ricker","ricker rotate45"),
              output_name =output_path + "wavelet_ricker.png");


conv_matrix1 = Post.seismic_conv_matrix(wavelet_ricker1, lt);
conv_matrix2 = Post.seismic_conv_matrix(wavelet_ricker2, lt);

PF.imshow(conv_matrix1,
              output_name =output_path + "conv_matrix1.png");
PF.imshow(conv_matrix2,
              output_name =output_path + "conv_matrix2.png");

'''generate reflectivity'''
ref_arr = module.zeros((lt, 1), dtype=module.float32);

positions_list                = [100, 110];
ref_arr[positions_list[0], 0] = +0.02;
ref_arr[positions_list[1], 0] = -0.02;



'''generate data'''
s1    = Post.apply_matrix_mul_last_and_first(conv_matrix1, ref_arr)
s2    = Post.apply_matrix_mul_last_and_first(conv_matrix2, ref_arr)

s1_with_noise = WR.add_noise_with_snr(s1, snr_db=snr_db)
s2_with_noise = WR.add_noise_with_snr(s2, snr_db=snr_db)


PF.plot_graph([s2, s2_with_noise],
              plot_number =2,
              legend_name =("s2","s2_with_noise"),
              output_name =output_path + "s2_with_noise.png");

PF.plot_graph([ref_arr, s2_with_noise],
              plot_number =2,
              legend_name =("ref","s2_with_noise"),
              output_name =output_path + "ref.png");

'''def forward and adjoint operators'''
data_term_forward = lambda x: Post.apply_matrix_mul_last_and_first(conv_matrix2,   x)
data_term_adjoint = lambda x: Post.apply_matrix_mul_last_and_first(conv_matrix2.T, x)


WR.dot_test(data_term_forward, data_term_adjoint, x=ref_arr );




# -------------------------------
# 5.train,  find optimal solution
# -------------------------------
num_episodes    = 1000;
batch_size      = 128;
gamma           = 0.95;
lr              = 1e-4;

'''
    In the early stage of the iteration, a larger exploration can be adopted. In the later stage, it is set to 0 and no random exploration is conducted
'''
epsilon_start   = 1;
epsilon_end     = 0;
epsilon_arr     = np.linspace(epsilon_start, epsilon_end,  num_episodes);

dqn_convergence = [100, 0.00001];


env             = ReflectivityEnv(lt, 
                                  envelope_list=[2, 4], 
                                  forward_op=data_term_forward, 
                                  obs=s2_with_noise, 
                                  train_snr=train_snr, 
                                  module=module);

fixed_state     = module.zeros(  (env.lt,), dtype=module.float32  );
fixed_state[positions_list[0]] =  0.01;
fixed_state[positions_list[1]] = -0.01;

state_dim       = env.state_dim;
action_dim      = env.action_dim;
buffer_capacity = state_dim*10;
reward_way      = 2;
random_type     = 0;

'''
    allocate network and share the parameter
'''
policy_net      = DQNNet(state_dim, action_dim).to(device);
target_net      = DQNNet(state_dim, action_dim).to(device);

target_net.load_state_dict(policy_net.state_dict());
optimizer       = optim.Adam(policy_net.parameters(), lr=lr);
replay_buffer   = ReplayBuffer(buffer_capacity);

update_target_loop  = state_dim//2
log_every_episode   = 4

rewards_history     = [];
loss_history        = [];

file_dict   = {}



for episode in range( num_episodes ):
    
    epsilon = epsilon_arr[episode];
    
    '''
        generate a random current_state
    '''
    
    current_state   = env.reset(
                                # fixed_state   =fixed_state,
                                
                                ref_min_max   =[-0.03, 0.03],
                                positions_list=positions_list,
                                );
    episode_reward  = 0;
    episode_loss    = 0;
    
    
    for inner_loop in range( state_dim*5 ):
        
        '''
            generate the valid_actions space and valid_action_indices
            \eplison behavior policy
        '''
        valid_actions, valid_action_indices = env.valid_actions(current_state);
        
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
                
                if module.__name__ in ['numpy', 'cupy']:
                    q_valid = module.array([q_values[i] for i in valid_action_indices]);
                elif module.__name__ == 'torch':
                    q_valid = module.tensor([q_values[i] for i in valid_action_indices]);
                
                best_idx     = module.argmax( q_valid ).item();
                
                action       = valid_actions[best_idx];
                action_idx   = valid_action_indices[best_idx];
                
        '''
            Store the experience samples generated by \pi_b
            ( s, a, r, s' )
            ( current_state, action_idx, reward, next_state )
        '''
        next_state, reward, done = env.step_and_reward( action, current_state,
                                                        relative_per=0.02
                                                      );
        
        episode_reward          += reward;
        
        
        replay_buffer.push(current_state, action_idx, reward, next_state, done);
        current_state   = next_state;
        
        if done:
            break;
        
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
            
            
            
        if inner_loop>= batch_size and inner_loop % update_target_loop == 0:
            target_net.load_state_dict( policy_net.state_dict() );
    
    
    rewards_history.append( episode_reward );
    loss_history.append(    episode_loss   );
    
    
    if episode % log_every_episode == 0:
        nonzero_indices = module.nonzero(current_state);
        nonzero_values  = current_state[nonzero_indices];
        
        file_dict['1'] = f"Episode         = {episode+1}";
        file_dict['2'] = f"inner_loop      = {inner_loop}";
        file_dict['3'] = f"env.snr2        = {env.snr2:.3f}";
        file_dict['4'] = f"episode_reward  = {episode_reward:.3f}";
        file_dict['5'] = f"nonzero_indices = {nonzero_indices}";
        file_dict['6'] = f"nonzero_values  = {nonzero_values}";
        
        for key, value in file_dict.items():
            write_txt(  log_file, value, print_bool=True );
        
        
        # ----- rewards_history -----
        plt.figure(figsize=(8, 5), dpi=200);
        plt.plot(rewards_history, linewidth=2);
        plt.xlabel('Episode', fontsize=14);
        plt.ylabel('Total reward per Episode', fontsize=14);
        plt.title('DQN Source Placement: Episode Reward', fontsize=16);
        plt.tight_layout();
        plt.savefig(output_path+f"{state_dim}-{action_dim}-rewards_history.png");
        plt.close();
        
        # ----- loss_history -----
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(loss_history, linewidth=2)
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Q-Value Loss', fontsize=14)
        plt.title('DQN Source Placement: Q-Value Loss', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path+f"{state_dim}-{action_dim}-loss_history.png")
        plt.close()
        
        
        
        
    if episode > dqn_convergence[0]:
        avg_reward_prev = np.mean(rewards_history[-dqn_convergence[0]:-1])
        if avg_reward_prev != 0:
            rel_change = abs(episode_reward - avg_reward_prev) / abs(avg_reward_prev)
            if rel_change < dqn_convergence[1]:
                
                file1 = f"\n Early stop at episode {episode+1}! Relative reward change < {dqn_convergence[1]}%";
                
                write_txt(  log_file, file1, print_bool=True );
                
                break;


def predict_best_state(env, state, policy_net, device, max_steps=1000):
    
    trajectory   = [state.copy() if hasattr(state, 'copy') else state.clone()];
    
    total_reward = 0.0;
    
    for _ in range(max_steps):
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0);
        
        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy()[0]
        
        valid_actions, valid_action_indices = env.valid_actions(state)
        if len(valid_actions) == 0:
            break
        
        q_valid     = np.array([q_values[i] for i in valid_action_indices])
        best_idx    = np.argmax(q_valid)
        action      = valid_actions[best_idx]
        next_state, reward, done = env.step_and_reward(action, state, relative_per=0.02)
        total_reward += reward
        state       = next_state
        trajectory.append(state.copy() if hasattr(state, 'copy') else state.clone())
        
        if done:
            break
    return state, total_reward, trajectory


def predict_best_state_function():
    
    current_state   = env.reset(
                                # fixed_state   =fixed_state,
                                
                                ref_min_max   =[-0.03, 0.03],
                                positions_list=positions_list,
                                );
    
    state, total_reward, trajectory = predict_best_state(env, current_state, 
                                                         policy_net, device,
                                                         max_steps=10000);
    
    ref_arr2 = WR.array_squeeze(ref_arr);
    ncc     = WR.ncc(ref_arr2, state, axis=0)
    
    PF.plot_graph([ref_arr2, state, ref_arr2-state],
                  plot_number =3,
                  legend_name =("ref",f"inv_ref,ncc={ncc:.2f}", "res"),
                  output_name =output_path + "inv_ref.png");
    
    nonzero_indices = module.nonzero(state);
    nonzero_values  = state[nonzero_indices];
    print(f"nonzero_indices ={nonzero_indices}");
    print(f"nonzero_values ={nonzero_values}");
    
predict_best_state_function();