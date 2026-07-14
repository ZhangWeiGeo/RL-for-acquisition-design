import numpy as np
import matplotlib.pyplot as plt
import os


'''
    environment:
        states,
        actions,
        rewards,
        dynamics of environment $p(s'|s, a)$ and $p(r|s, a)$,
    agent:
        policy,
        optimal policy,
'''
class  TrapGridWorld:
    def __init__(self):
        self.size  = 5
        self.start = (0, 0)
        self.goal  = (4, 4)
        self.traps = [(1, 1), (1, 3), (2, 3), (3, 0)]
        '''up, down, left and right'''
        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]
        self.n_states = self.size * self.size
        self.n_actions = len(self.actions)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.state_num(self.pos)
    
    '''flatten the state vector as a unique point value'''
    def state_num(self, pos):
        return pos[0] * self.size + pos[1]
    
    '''return position_idx as (row, col)'''
    def state_num_reverse(self, vector_value):
        i = vector_value // self.size
        j = vector_value % self.size
        return (i, j)
    
    def state_to_vec(self,s):
        vec    = np.zeros(self.n_states, dtype=np.float32)
        vec[s] = 1.0
        return vec
    
    '''perform action and get reward'''
    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.pos
        '''state must be in [0,self.size-1]'''
        nx, ny = min(self.size-1, max(0, x+dx)), min(self.size-1, max(0, y+dy))
        self.pos = (nx, ny)
        
        if self.pos in self.traps:
            '''move in to trap'''
            return self.state_num(self.pos), -10, True, {};
        elif self.pos == self.goal:
            '''get the gold'''
            return self.state_num(self.pos), +10, True, {};
        else:
            '''get a normal step'''
            return self.state_num(self.pos), -1, False, {}
    
    '''
        Dynamics of environment: $p(s'|s, a)$ and $p(r|s, a)$
        
        create P[state][action] = [(prob, next_state, reward, done), ...]
            In other words, we need to generate four arrays of [state][action]
            1: p(s'|s,a)=1
            2: the idx number of s' 
            3: p(r|s,a),  r=reward
            4: is or not done
            see Algorithm 4.2: Policy iteration algorithm
    '''
    def build_transition_table(self):
        
        P = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]
        for i in range(self.size):
            for j in range(self.size):
                s = self.state_num((i, j));
                for a, (dx, dy) in enumerate(self.actions):
                    
                    ni, nj = min(self.size-1, max(0, i+dx)), min(self.size-1, max(0, j+dy));
                    pos_vector = self.state_num( (ni, nj) );
                    pos = (ni, nj);
                    
                    # reward
                    if pos in self.traps:
                        reward = -10;
                        done = True;
                    elif pos == self.goal:
                        reward = 10;
                        done = True;
                    else:
                        reward = -1;
                        done = False;
                    '''determined transation 1.0'''
                    P[s][a] = [(1.0, pos_vector, reward, done)]
        return P
    
    
    def render(self):
        ''' for plot or print'''
        grid = np.full((self.size, self.size), 'Road')
        
        for hx, hy in self.traps:
            grid[hx, hy] = 'Trap'
        
        grid[self.goal]  = 'Goal'
        grid[self.pos]   = 'Agent'
        for row in grid:
            print(' '.join(row))
        print()


def demo_environment(output_path, action_list=[], name="episode-0"):
    '''
        perform multiple steps test
    '''
    print("demo_environment: move one step")
    
    direction_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    env   =  TrapGridWorld()
    state = env.reset()
    env.render()
    done = False
    steps = 0;
    cumulative_reward= 0;
    plot_env(env, output_name=output_path+f"{name}-0.png", plt_title=f"Initial state");
    
    reward_list = [0,];
    state_list  = [env.state_num_reverse(state)]
    
    if len(action_list)==0:
        total_steps = 10;
    else:
        total_steps = len(action_list);
    
    while not done and steps < total_steps:
        if len(action_list)==0:
            action = np.random.choice(4)
        else:
            action = action_list[steps]
        next_state, reward, done, _ = env.step(action)
        
        cumulative_reward+=reward;
        reward_list.append( reward  );
        state_list.append(  env.state_num_reverse(next_state)  )
        
        plot_env(env, output_name=output_path+f"{name}-{steps+1}.png", plt_title=f"Step:{steps}, {direction_dict[action]}, Immediate reward={reward}, cumulative reward={cumulative_reward}");
        steps += 1
        
    plot_episode(env, state_list, 
                     action_list, 
                     reward_list, 
                     output_name=output_path+f"{name}.png", 
                     plt_title=f"{name},");

'''
    value iteration and policy iteration
    They are the standard algorithms for solving Markov decision Processes (MDP).
    Dynamic Programming, DP
'''
def value_iteration(env, gamma=0.95, theta=1e-6, max_iterations=1000):
    '''
        p59-60
        see Algorithm 4.1: Policy iteration algorithm
    '''
    n_states  = env.n_states;
    n_actions = env.n_actions;
    P         = env.build_transition_table();

    v_table = np.zeros(n_states)
    q_table = np.zeros((n_states, n_actions))
    policy  = np.zeros(n_states, dtype=int)  # save the greed policy
    q_table_history = []
    
    for iteration in range(max_iterations):
        
        q_table_history.append(  q_table.copy()  )
        
        '''0 q-value:'''
        delta = 0
        for s in range(n_states):

            for a in range(n_actions):
                q_sa = 0
                for prob, next_state, reward, done in P[s][a]:
                    q_sa += prob * (reward + gamma * v_table[next_state])
                q_table[s, a] = q_sa
        
        '''1 Policy update:
               Greedy policy: for each state, pick the action with max Q-value
        '''
        policy = np.argmax(q_table, axis=1)# shape [n_states]
        
        '''2. Value update:'''
        v_table_new = np.max(q_table, axis=1);
        delta       = np.max(np.abs(v_table_new - v_table))
        v_table     = v_table_new
        
        if delta < theta:
            print(f"Value iteration converged at iteration {iteration}")
            break
    
    return v_table, q_table, policy, q_table_history


def policy_iteration(env, gamma=0.95, theta=1e-6, max_iterations=1000):
    '''
        p65-66
        see Algorithm 4.2: Policy iteration algorithm
    '''
    n_states  = env.n_states
    n_actions = env.n_actions
    P         = env.build_transition_table();
    
    '''
        Initialization policy: Each state is uniformly random
    '''
    policy  = np.ones((n_states, n_actions)) / n_actions;
    v_table = np.zeros(n_states);
    q_table = np.zeros((n_states, n_actions));
    q_table_history = []

    for iteration in range(max_iterations):
        
        q_table_history.append( q_table.copy() )
        
        '''
            1 the policy evaluation step,  update v_table
              perform the fixed-point iteration or truncated fixed-point iteration
        '''
        while True:
            delta = 0
            for s in range(n_states):
                v = v_table[s]
                new_v = 0
                for a in range(n_actions):
                    q_sa = 0
                    for prob, next_state, reward, done in P[s][a]:
                        '''p66:'''
                        q_sa += prob * (reward + gamma * v_table[next_state])
                    
                    new_v += policy[s, a] * q_sa
                    q_table[s, a] = q_sa  # update q table
                delta = max(delta, abs(v - new_v))
                v_table[s] = new_v
            if delta < theta:
                break

        '''2 the policy improvement step:
             Let each state only select the action with the maximum q
        '''
        policy_stable = True
        for s in range(n_states):
            old_action  = np.argmax(policy[s])
            best_action = np.argmax(q_table[s])
            '''new policy'''
            policy[s] = np.eye(n_actions)[best_action]
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            print(f"Policy iteration converged at iteration {iteration}")
            break
    
    return v_table, q_table, np.argmax(policy,1), q_table_history

'''
P143 
If an algorithm is on-policy, then it can be implemented in an online fashion, but cannot use pre-collected data generated by other policies. 
If an algorithm is off-policy, then it can be implemented in either an online or offline fashion.

My understanding:
    If you need to generate the new samples based on the updated policy, then it is on-policy
    Informally, the value iteration and policy iteration can be considered as on-policy version.
'''
def mc_epsilon_greedy(env, 
                      n_episodes=5000, 
                      tmax_length=500,
                      
                      gamma=0.95, 
                      epsilon_max=0.1,
                      epsilon_min=0.0,
                      theta=1e-6):
    '''
        Monte Carlo (MC) algorithms with epsilon_greedy
        Algorithm 5.3  and (also see 5.2 for Exploring Starts)
        see p 91
        
        
        model-free, on-policy, we need save the samples of enviorment
        (we can use the difference method, then on polciy, we do not save the samples?)
        
        problem 1: why T_max to T_min ? see Algorithm 5.2
        
        A trade-off exploration and exploitation:
            Here, exploration means that the policy can possibly take as many actions as possible. 
            In this way, all the actions can be visited and evaluated well. 
            
            Exploitation means that the improved policy should take the greedy action that has the greatest action value. 
            
            However, since the action values obtained at the current moment may not be accurate due to insufficient exploration, we should keep exploring while conducting exploitation to avoid missing optimal actions. ( P92, 5.5 )
    '''
    
    n_states  = env.n_states
    n_actions = env.n_actions

    
    q_table = np.zeros((n_states, n_actions))
    Returns = np.zeros((n_states, n_actions))
    Num     = np.zeros((n_states, n_actions))
    
    '''Initialize policy to uniform random'''
    policy = np.ones((n_states, n_actions)) / n_actions

    for episode in range(n_episodes):
        
        epsilon = epsilon_min + episode * (epsilon_max - epsilon_min)/n_episodes;
        
        '''1. Generate one episode ---'''
        state = env.reset()
        episode_data = []

        done = False
        t_loop=0
        while not done:
            '''Sample action according to current ε-greedy policy'''
            action = np.random.choice(n_actions, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            episode_data.append( (state, action, reward) )
            state = next_state
            t_loop+=1;
            if t_loop > tmax_length:
                done=True;

        '''2. Monte Carlo evaluation (backward) ---'''
        G = 0
        visited = set()
        for t in reversed( range( len(episode_data) ) ):
            s, a, r = episode_data[t]
            G = gamma * G + r
            '''
                First-visit MC
                This is quite understandable because we only perform mean estimation for the last visit, which why we visit the samples from the end time of episode
            '''
            if (s, a) not in visited:
                Returns[s, a] += G
                Num[s, a]     += 1
                q_table[s, a] = Returns[s, a] / Num[s, a]
                visited.add( (s, a) )
                '''estimate the mean of G_t'''
        
        '''3. Policy improvement (ε-greedy) ---'''
        for s in range(n_states):
            a_star = np.argmax(q_table[s])
            for a in range(n_actions):
                if a == a_star:
                    policy[s, a] = 1 - epsilon + epsilon / n_actions
                else:
                    policy[s, a] = epsilon / n_actions
        
        # q_table_last = q_table.copy()
        # delta = np.max(np.abs(q_table - q_table_last))
        # if delta < theta:
        #     print('mc_epsilon_greedy: Converged at episode', episode)
        #     break
        
    return q_table, np.argmax(policy, axis=1)


def train_q_learning_on_policy(env, 
                     n_episodes=500, 
                     tmax_length=500,
                     
                     alpha=0.8, 
                     gamma=0.95, 
                     epsilon_max=0.1,
                     epsilon_min=0.0,
                     ):
    '''
        on policy
        see 7.2,  p143
        q tables
        model-free, on-policy, we do not need save the samples of enviorment
        (we use the difference method)
    '''
    n_states  = env.n_states
    n_actions = env.n_actions

    q_table = np.zeros((n_states, n_actions))
    reward_list = []

    for episode in range(n_episodes):
        
        state       = env.reset();
        done        = False;
        cumulative_reward= 0;
        t_loop      =0;
        epsilon = epsilon_min + episode * (epsilon_max - epsilon_min)/n_episodes;
        
        while not done:
            '''
            1. Update policy:
                ε-greed policy
                larger ε will lead to larger exploration
            '''
            if np.random.rand() < epsilon:
                action = np.random.choice(env.n_actions);
            else:
                '''greed policy'''
                action = np.argmax(q_table[state]);
            
            next_state, reward, done, _ = env.step(action);
            cumulative_reward += reward;
            
            
            t_loop+=1;
            if t_loop > tmax_length:
                done=True;
            
            '''
            2. Q-value update:
                reward: r
            '''
            best_next_action = np.argmax(q_table[next_state])
            td_target   = reward + gamma * q_table[next_state, best_next_action]
            td_error    =  q_table[state, action] - td_target
            q_table[state, action] -= alpha * td_error

            state = next_state

        reward_list.append(cumulative_reward)
    return q_table, reward_list


def train_q_learning_off_policy(env, 
                     n_episodes=500, 
                     tmax_length=500,
                     
                     alpha=0.8, 
                     gamma=0.95, 
                     epsilon_max=0.1,
                     epsilon_min=0.0,
                     ):
    '''
        off policy
        see 7.3,  p143
        q tables
        model-free, you do not need save the samples of enviorment
    '''
    n_states  = env.n_states
    n_actions = env.n_actions

    q_table = np.zeros((n_states, n_actions))
    reward_list = []
    
    '''Initialization strategy: Each state is uniformly random'''
    policy  = np.ones((n_states, n_actions)) / n_actions;

    for episode in range(n_episodes):
        
        state       = env.reset();
        done        = False;
        cumulative_reward= 0;
        t_loop      =0;
        epsilon = epsilon_min + episode * (epsilon_max - epsilon_min)/n_episodes;
        
        while not done:
            
            '''0. Sample action according to behavior policy'''
            action = np.random.choice(n_actions, p=policy[state])
            next_state, reward, done, _ = env.step(action)
            
            t_loop+=1;
            if t_loop > tmax_length:
                done=True;
            
            '''1. update Q value'''
            best_next_action = np.argmax(q_table[next_state])
            td_target   = reward + gamma * q_table[next_state, best_next_action]
            td_error    =  q_table[state, action] - td_target
            q_table[state, action] -= alpha * td_error

            state = next_state
            cumulative_reward+=reward;
            
            '''2. Update target policy'''
            target_policy = np.argmax(q_table, axis=1);
        
        
        reward_list.append(cumulative_reward)
        
    return q_table, reward_list


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn(env, n_episodes=1000, 
                   tmax_length=500, 
                   gamma=0.95, 
                   epsilon_max=0.1, 
                   epsilon_min=0.01, 
                   alpha=1e-3, 
                   memory_size=10000, 
                   batch_size=64, 
                   target_update=10, 
                   device='cpu'):
    
    n_states  = env.n_states
    n_actions = env.n_actions
    
    q_net    = QNet(n_states, n_actions).to(device)
    q_target = QNet(n_states, n_actions).to(device)
    q_target.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=alpha)
    memory    = deque(maxlen=memory_size)

    epsilon       = epsilon_max
    epsilon_decay = (epsilon_max - epsilon_min) / n_episodes

    reward_list = []
    for episode in range(n_episodes):
        
        state = env.reset()
        cumulative_reward = 0
        done = False
        t_loop = 0

        while not done and t_loop < tmax_length:
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                state_vec = torch.tensor(env.state_to_vec(state)).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = q_net(state_vec)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            cumulative_reward += reward
            state = next_state
            t_loop += 1

            '''DQN experience replay'''
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                batch_s, batch_a, batch_r, batch_ns, batch_done = zip(*batch)

                batch_s = torch.tensor([env.state_to_vec(s) for s in batch_s]).to(device)
                batch_a = torch.tensor(batch_a).unsqueeze(1).to(device)
                batch_r = torch.tensor(batch_r, dtype=torch.float32).unsqueeze(1).to(device)
                batch_ns = torch.tensor([env.state_to_vec(s) for s in batch_ns]).to(device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

                q_pred = q_net(batch_s).gather(1, batch_a)
                with torch.no_grad():
                    q_next = q_target(batch_ns).max(1, keepdim=True)[0]
                    q_target_val = batch_r + gamma * q_next * (1 - batch_done)

                loss = nn.MSELoss()(q_pred, q_target_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        reward_list.append(cumulative_reward)
        epsilon = max(epsilon - epsilon_decay, epsilon_min)

        '''update target network'''
        if episode % target_update == 0:
            q_target.load_state_dict(q_net.state_dict())

        if (episode+1) % 100 == 0:
            print(f"Deep Q-learning, Episode {episode+1}, reward: {cumulative_reward:.2f}")

    return q_net, reward_list


'''plot function'''
def plot_episode(env, state_list, action_list, reward_list, 
                 output_name="episode.png", 
                 plt_title="Episode Trajectory", 
                 title_fontsize=20, 
                 cell_fontsize=22, 
                 arrow_fontsize=28):
    """
        env        : 
        state_list : (0[(0,0), (1,0), ...])
        action_list: ([1, 3, ...])
        reward_list:   reward at each step
    """
    color_dict = {
        'Road':  [0.7, 0.9, 1.0],  # blue
        'Trap':  [0.9, 0.3, 0.3],  # red
        'Goal':  [0.3, 0.9, 0.3],  # green
        'Agent': [1.0, 0.9, 0.3],  # yellow
        'Visited': [0.9, 0.8, 0.4],
        'Start': [1.0, 0.6, 0.2],
    }
    direction_dict = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    grid = np.full((env.size, env.size), 'Road', dtype=object)
    for hx, hy in env.traps:
        grid[hx, hy] = 'Trap'
    grid[env.goal] = 'Goal'
    for pos in state_list:
        if grid[pos] not in ['Trap', 'Goal']:
            grid[pos] = 'Visited'
    grid[env.start] = 'Start'
    grid[env.goal] = 'Goal'

    color_grid = np.array([[color_dict[cell] for cell in row] for row in grid])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(color_grid, interpolation='none')


    #mark step number and arrow
    for step, pos in enumerate(state_list[:-1]):
        i, j = pos
        action = action_list[step]
        arrow = direction_dict[action]
        ax.text(j, i, f"{arrow}\n{step}", ha='center', va='center', fontsize=arrow_fontsize, fontweight='bold', color='black')


    cumulative_reward = sum( reward_list );
    # Trap/Goal/Start 
    for i in range(env.size):
        for j in range(env.size):
            if grid[i, j] == 'Trap':
                ax.text(j, i, 'Trap', ha='center', va='center', fontsize=cell_fontsize, fontweight='bold', color='black')
            elif grid[i, j] == 'Goal':
                ax.text(j, i, 'Goal', ha='center', va='center', fontsize=cell_fontsize, fontweight='bold', color='black')
            elif grid[i, j] == 'Start':
                ax.text(j, i, 'Start', ha='center', va='center', fontsize=cell_fontsize, fontweight='bold', color='black')
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(plt_title+f" cumulative reward={cumulative_reward}", fontsize=title_fontsize, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close(fig)



def plot_env(env, output_name="state.png", 
             plt_title="", 
             title_fontsize=20, cell_fontsize=24):
    """env:  size、traps、goal、pos """
    color_dict = {
        'Road':  [0.7, 0.9, 1.0],  # blue
        'Trap':  [0.9, 0.3, 0.3],  # red
        'Goal':  [0.3, 0.9, 0.3],  # green
        'Agent': [1.0, 0.9, 0.3],  # yellow
    }
    grid = np.full((env.size, env.size), 'Road', dtype=object)
    for hx, hy in env.traps:
        grid[hx, hy] = 'Trap'
    grid[env.goal] = 'Goal'
    grid[env.pos]  = 'Agent'

    color_grid = np.array([[color_dict[cell] for cell in row] for row in grid])

    fig, ax = plt.subplots(figsize=(10, 10))  # 
    ax.imshow(color_grid, interpolation='none')

    # add text
    for i in range(env.size):
        for j in range(env.size):
            label = grid[i, j]
            ax.text(j, i, label, ha='center', va='center', fontsize=cell_fontsize, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if plt_title:
        plt.title(plt_title, fontsize=title_fontsize, pad=20, fontweight='bold');
    plt.tight_layout();
    plt.savefig(output_name);
    plt.close(fig);

def plot_qtable_policy(env, q_table, 
                       output_name="Q_policy.png", 
                       title="Greedy Policy and Max Q-value", 
                       title_fontsize=20, 
                       cell_fontsize=18,
                       arrow_fontsize=32):
    
    direction_dict = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    color_dict = {
        'Road':  [0.7, 0.9, 1.0],
        'Trap':  [0.9, 0.3, 0.3],
        'Goal':  [0.3, 0.9, 0.3],
        'Agent': [1.0, 0.9, 0.3],
    }
    grid = np.full((env.size, env.size), 'Road', dtype=object)
    for hx, hy in env.traps:
        grid[hx, hy] = 'Trap'
    grid[env.goal] = 'Goal'

    color_grid = np.array([[color_dict[cell] for cell in row] for row in grid])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(color_grid, interpolation='none')

    for i in range(env.size):
        for j in range(env.size):
            state = env.state_num((i, j))
            if grid[i, j] == 'Trap':
                ax.text(j, i, 'Trap', ha='center', va='center', fontsize=cell_fontsize, fontweight='bold', color='black')
            elif grid[i, j] == 'Goal':
                ax.text(j, i, 'Goal', ha='center', va='center', fontsize=cell_fontsize, fontweight='bold', color='black')
            else:
                best_action = np.argmax(q_table[state])
                best_q = q_table[state, best_action]
                arrow = direction_dict[best_action]
                # arrows and q value
                ax.text(j, i, arrow, ha='center', va='center', fontsize=arrow_fontsize, fontweight='bold', color='black')
                ax.text(j, i+0.28, f"{best_q:.1f}", ha='center', va='center', fontsize=cell_fontsize, fontweight='bold', color='black')

    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title, fontsize=title_fontsize, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close(fig)


''' main'''
if __name__ == "__main__":
    
    epsilon_max =1
    
    output_path = f"./Q-learning-{epsilon_max}/";
    try:
        os.rmdir(output_path)
    except OSError:
        pass
    os.makedirs(output_path, exist_ok=True);
    
    '''
        demo test for episode, some basic concepts
    '''
    direction_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'};
    
    os.makedirs(output_path+"episode/", exist_ok=True);
    demo_environment( output_path+"episode/", action_list=[3, 3, 1, 1, 1, 3, 0], name="episode-0" )
    demo_environment( output_path+"episode/", action_list=[1, 1, 1, 3, 0,], name="episode-1" )
    demo_environment( output_path+"episode/", action_list=[1, 1, 3, 1, 3, 3, 3, 1], name="episode-2" )
    
    
    '''
        build enviorment
    '''
    env =  TrapGridWorld()
    
    
    ''' value_iteration '''
    v_table1, q_table1, policy1, q_table1_history = value_iteration(env, gamma=0.95, theta=1e-6, max_iterations=1000);
    
    plot_qtable_policy(env, q_table1, title="Value_iteration: Greedy Policy and Max Q-value", output_name=output_path + "value_iteration_policy.png");
    
    for iteration, q_table_tmp in enumerate( q_table1_history ):
        if iteration % 2==0 and iteration<10:
            plot_qtable_policy(env,  q_table_tmp, title=f"Value_iteration:{iteration}: Greedy Policy and Max Q-value", output_name=output_path + f"value_iteration_policy-{iteration}.png");
    
    
    
    ''' policy_iteration '''
    v_table2, q_table2, policy2, q_table2_history = policy_iteration(env, gamma=0.95, theta=1e-6, max_iterations=1000);
    
    plot_qtable_policy(env, q_table2, title="Policy_iteration: Greedy Policy and Max Q-value", output_name=output_path + "policy_iteration_policy.png");
    
    for iteration, q_table_tmp in enumerate( q_table2_history ):
        
        plot_qtable_policy(env,  q_table_tmp, title=f"Policy_iteration:{iteration}: Greedy Policy and Max Q-value", output_name=output_path + f"policy_iteration_policy-{iteration}.png");
    
    
    '''
        mc_epsilon_greedy, on policy
    '''
    mc_q_table, mc_policy = mc_epsilon_greedy(env, 
                                          n_episodes=10000, 
                                          tmax_length=20000,
                                          gamma=0.95,
                                          epsilon_max=1, 
                                          epsilon_min=0);
    
    plot_qtable_policy(env, mc_q_table, title="MC -Greedy: Greedy Policy and Max Q-value", output_name=output_path + "MC_policy.png");
    
    

    '''
        Q-learning on policy
    '''
    q_learning_q_table_on, q_learning_on_reward_list = train_q_learning_on_policy(env, 
                                            n_episodes=500, 
                                            tmax_length=500,
                                            alpha=0.8, 
                                            gamma=0.95, 
                                            epsilon_max=epsilon_max, 
                                            epsilon_min=0);
    
    plot_qtable_policy(env, q_learning_q_table_on, title="Q-learning on policy: Greedy Policy and Max Q-value", output_name=output_path + "q_learning_on_policy.png");
    
    
    
    # # 3. Plot Q-table
    # plt.figure(figsize=(10, 6))
    # plt.imshow(q_table, cmap='viridis', aspect='auto')
    # plt.colorbar(label="Q-value")
    # plt.xlabel("Action (0:up, 1:down, 2:left, 3:right)", fontsize=14)
    # plt.ylabel("State (0~15)", fontsize=14)
    # plt.title("Q-table after Training", fontsize=18)
    # plt.tight_layout()
    # plt.savefig(output_path + "Q_table.png")
    # plt.close()
    
    # Plot reward
    plt.figure(figsize=(10, 4))
    plt.plot(q_learning_on_reward_list, label="Episode Reward")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.title("Reward per Episode during Training", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + "q_learning_reward.png")
    plt.close()
    
    '''
        Q-learning off policy
    '''
    q_learning_q_table_off, q_learning_off_reward_list = train_q_learning_off_policy(env, 
                                            n_episodes=5000, 
                                            tmax_length=20000,
                                            alpha=0.8, 
                                            gamma=0.95, 
                                            epsilon_max=epsilon_max, 
                                            epsilon_min=0);
    
    plot_qtable_policy(env, q_learning_q_table_off, title="Q-learning off policy: Greedy Policy and Max Q-value", output_name=output_path + "q_learning_off_policy.png");
    
    
    
    '''deep Q learning'''
    q_net, deep_q_learning_on_reward_list = train_dqn(env, 
                                                   n_episodes=200, 
                                                   tmax_length=500, );
    
    all_states        = np.eye(env.n_states, dtype=np.float32) # shape (n_states, n_states)
    all_states_tensor = torch.tensor(all_states)              # 

    with torch.no_grad():
        q_table_pred = q_net(all_states_tensor)              # shape (n_states, n_actions)
        q_table_numpy = q_table_pred.cpu().numpy()           # 
    
    plot_qtable_policy(env, q_table_numpy, title="Deep Q-learning: Greedy Policy and Max Q-value", output_name=output_path + "deep_q_learning.png");
    
    
    plt.figure(figsize=(10, 4))
    plt.plot(deep_q_learning_on_reward_list, label="Episode Reward")
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.title("Reward per Episode during Training", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + "deep_q_learning_reward.png")
    plt.close()
    
    
    
    '''
        Why on-policy is faster? On-policy learning often leads to faster and more stable convergence because it always updates based on its own, current behavior. 
        
        Why do we still need off-policy Q-learning?
        1. Data Efficiency: Off-policy algorithms can learn from data generated by any behavior policy, allowing reuse of past experiences or data collected from different strategies.
        
        2. Safe Exploration: Off-policy methods can train a target policy while exploring with a different, potentially safer, behavior policy.
        
        3. Batch/Offline Learning: Off-policy Q-learning is suitable for scenarios where interaction with the environment is limited or expensive, and learning must be performed using pre-collected datasets.
        
        4. Imitation and Transfer Learning: Enables learning from demonstrations or policies collected from other agents or experts.
        
        5. Theoretical Foundation: The original Q-learning algorithm is inherently off-policy and guarantees convergence under broad conditions.
    '''