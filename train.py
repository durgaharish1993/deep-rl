from lib import wrapper
from lib import dqn_model

import argparse
import time 
import numpy as np 
import collections 

import torch 
import torch.nn as nn 

import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-V4"
MEAN_REWARD_BOUND = 19.0 

# Bellman Approximation 
GAMMA = 0.99

BATCH_SIZE = 32 # Sampled from Replay Buffer 
REPLAY_SIZE = 10000 # Maximum capacity of buffer 
REPLAY_START_SIZE = 10000 # Count of frames before starting training 
LEARNING_RATE = 1e-4 # Learning Rate 
SYNC_TARGET_FRAMES = 1000 # Syncing weights between training network and target network for next state approx.

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0 # All actions to follow random 
EPSILON_FINAL = 0.01 # 1% of steps taken random action 

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), \
                np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), \
                np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env 
        self.exp_buffer = exp_buffer

        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0 

    @torch.no_grad
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None 

        if np.random.random() < epsilon :
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device=device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        new_state, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward +=reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
    


# Q(s,a) = r + gamma (max a' Q(s',a'))
def calc_loss(batch,net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch 

    states_v = torch.tensor(np.array(states, copy=False)).to(device=device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device=device)
    actions_v = torch.tensor(actions).to(device=device)
    rewards_v = torch.tensor(rewards).to(device=device)
    done_mask = torch.BoolTensor(dones).to(device=device)
    # (B, a_dim) -> (B, 1) ## Selecting actions from actions_v (choosing those values)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0 # For the state where next state is done (there is next state value)

    next_state_values = next_state_values.detach() # This prevents to update gradients from flowing to Target NN

    expected_state_action_values = rewards_v + GAMMA * next_state_values

    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the Environment, default="+DEFAULT_ENV_NAME)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrapper.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device=device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device=device)


    writer = SummaryWriter(comment="-" + args.env)
    print(net)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0 
    ts_frame = 0 
    ts = time.time()
    best_m_reward = None 

    while True:
        frame_idx +=1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)

            ts_frame = frame_idx
            m_reward = np.mean(total_rewards[-100:])

            print("%d : done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (frame_idx, len(total_rewards), m_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            

















