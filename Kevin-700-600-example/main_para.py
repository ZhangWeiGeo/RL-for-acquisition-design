
import sys
import os

parameters_arr = sys.argv[:];    # parameters_arr = sys.argv[1:];
if len(parameters_arr) ==1:
    train_bool            = False
    predict_bool          = True;
    shot_num              = 70;
    
    normalized_bool       = False;
    num_episodes          = 800;
    num_each_episodes     = 10000;
    update_target_loop    = 5000;
    
    max_predict_steps     = 40000;
    
    log_every_episode     = 100;
    batch_size            = 128
    buffer_capacity       = 50000
    
    epsilon_start         = 0.99;
    epsilon_end           = 0.03;
    
    reward_way            = 2;
    random_type           = 0;
else:
    train_bool            = eval(parameters_arr[1])
    predict_bool          = eval(parameters_arr[2])
    shot_num              = int(parameters_arr[3])
    
    normalized_bool       = eval(parameters_arr[4])
    num_episodes          = int(parameters_arr[5])
    num_each_episodes     = int(parameters_arr[6])
    update_target_loop    = int(parameters_arr[7])
    
    max_predict_steps     = int(parameters_arr[8])
    
    log_every_episode     = int(parameters_arr[9])
    batch_size            = int(parameters_arr[10])
    buffer_capacity       = int(parameters_arr[11])
    
    epsilon_start         = float(parameters_arr[12])
    epsilon_end           = float(parameters_arr[13])
    
    reward_way            = int(parameters_arr[14])
    random_type           = int(parameters_arr[15])



if not train_bool and predict_bool:
    policy_net_file = ""
else:
    policy_net_file = ""


if update_target_loop>=num_each_episodes:
    sys.exit(f"{update_target_loop}>={num_each_episodes}, update_target_loop>=num_each_episodes")


local_vars = locals();

sys_info_dict         = {}
sys_info_dict['code'] = f"code:{parameters_arr[0]}"

for key, value in list(local_vars.items()): 
    if isinstance(value, (int, float, bool, list, str)):
        if key not in ["sys_info_dict", "parameters_arr"]:  
            sys_info_dict[key] = f"{value} (type: {type(value).__name__})"

for key, value in sys_info_dict.items():
    print(f"{key}: {value}\n");