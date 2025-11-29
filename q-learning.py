import gymnasium as gym
import collections

from torch.utils.tensorboard import SummaryWriter

# Environment
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9 
ALPHA = 0.2 
TEST_EPISODES = 20 

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state,_ = self.env.reset()
        # Tabular Q-Learning [[Q(s,a) = r + gamma * (max_a' Q(s',a'))]] 
        # Q-iteration  Q(s,a) = (1-alpha) * Q(s,a) + alpha * ( r + gamma * (max_a' Q(s',a')))
        self.values = collections.defaultdict(float)

    
    def sample_env(self):
        '''
        '''
        #Random Action 
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.state = self.env.reset()[0] if is_done else new_state
        #(s, a, r, s')
        return old_state, action, reward, new_state
    
    def best_value_and_action(self,state):
        best_value, best_action = None, None 
        for action in range(self.env.action_space.n):
            action_value = self.values[(state,action)]
            if best_value is None or action_value > best_value :
                best_value = action_value
                best_action = action
        return best_value, best_action
    
    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v 
        old_v = self.values[(s,a)]
        self.values[(s,a)] = old_v * (1-ALPHA) + new_v * ALPHA

    
    def play_episode(self, env):
        total_reward = 0.0 
        state,_ = env.reset()
        while True:
            
            _, action = self.best_value_and_action(state=state)
            new_state, reward,  terminated, truncated, _ = env.step(action)
            is_done = terminated or truncated
            total_reward +=reward 
            if is_done:
                break 
            state = new_state

        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-q-learning')

    iter_no =0 
    best_reward = 0.0
    while True:
        
        iter_no += 1
        (s,a,r,next_s) = agent.sample_env()
        agent.value_update(s=s, a=a, r=r, next_s=next_s)
        
        '''
        Testing the value with test episodes 
        '''
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(env=test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward 
        
        if reward > 0.8:
            print("Soved in %d iterations!" % iter_no)
            break 

    writer.close()




