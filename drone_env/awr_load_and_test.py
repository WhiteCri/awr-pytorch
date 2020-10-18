import random
from collections import deque

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from mlagents.envs import UnityEnvironment
from array import array
import datetime

from model import *

#record
record_path = './drone_playing_record.tw'

#model
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
model_path = 'drone_agent_awr_20201007-13-08-35.pt'
model_path = 'drone_agent_awr_advanced_20201007-22-21-54.pt'

# unity ml-agent
env_name = 'C:/Users/User/Desktop/unity_projects/drone_tw/Drone/Drone'
env = UnityEnvironment(file_name=env_name)
default_brain = env.brain_names[0]
train_mode = False
train_interval = 1
start_interval = 30
test_interval = 1
save_interval = 30
episode_terminate = 500

continuous = True

use_cuda = False
use_noisy_net = False
batch_size = 256
num_sample = 2048
critic_update_iter = 500
actor_update_iter = 1000
iteration = 100000
max_replay = 50000

gamma = 0.9
lam = 0.95
beta = 0.05
max_weight = 20.0
use_gae = True

state_size = 9
action_size = 3


class BinaryReader():
    def __init__(self, filename):
        self.record = open(filename, 'rb')
        self.tot_len = state_size + action_size + 1 + state_size + 1

    def read(self):
        self.ary = array('f')
        try:
            self.ary.fromfile(self.record, self.tot_len)
        except EOFError:
            return False, ()
        ary_python = self.ary.tolist()
        idx = 0
        state = ary_python[idx: idx + state_size]
        idx = idx + state_size
        action = ary_python[idx: idx + action_size]
        idx = idx + action_size
        reward = ary_python[idx: idx + 1][0]
        idx = idx + 1
        next_state = ary_python[idx: idx + state_size]
        idx = idx + state_size
        done = ary_python[idx: idx + 1][0]

        return True, (state, action, reward, next_state, done)


class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            gamma,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            use_continuous=False):
        self.model = BaseActorCriticNetwork(
            input_size, output_size, use_noisy_net, use_continuous=use_continuous)
        self.continuous_agent = use_continuous

        self.output_size = output_size
        self.input_size = input_size
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae

        self.actor_optimizer = optim.SGD(self.model.actor.parameters(),
                                          lr=0.00005, momentum=0.9)
        self.critic_optimizer = optim.SGD(self.model.critic.parameters(),
                                           lr=0.0001, momentum=0.9)
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.model = self.model.to(self.device)

    def get_action(self, state):
        # state = torch.Tensor(state).to(self.device).reshape(1,-1)
        # state = state.float()
        state = torch.tensor(state).float().reshape(1, -1)
        policy, value = self.model(state)

        if self.continuous_agent:
            action = policy.sample().numpy().reshape(-1)
        else:
            policy = F.softmax(policy, dim=-1).data.cpu().numpy()
            action = np.random.choice(np.arange(self.output_size), p=policy[0])

        return action

    def train_model(self, s_batch, action_batch, reward_batch, n_s_batch, done_batch):
        s_batch = np.array(s_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)

        data_len = len(s_batch)
        mse = nn.MSELoss()

        # update critic
        self.critic_optimizer.zero_grad()
        cur_value = self.model.critic(torch.FloatTensor(s_batch))
        #print('Before opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, _ = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        # discounted_reward = (discounted_reward - discounted_reward.mean())/(discounted_reward.std() + 1e-8)
        for _ in range(critic_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            sample_value = self.model.critic(torch.FloatTensor(s_batch[sample_idx]))
            if (torch.sum(torch.isnan(sample_value)) > 0):
                print('NaN in value prediction')
                input()
            critic_loss = mse(sample_value.squeeze(), torch.FloatTensor(discounted_reward[sample_idx]))
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        # update actor
        cur_value = self.model.critic(torch.FloatTensor(s_batch))
        #print('After opt - Value has nan: {}'.format(torch.sum(torch.isnan(cur_value))))
        discounted_reward, adv = discount_return(reward_batch, done_batch, cur_value.cpu().detach().numpy())
        #print('Advantage has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(adv).float()))))
        #print('Returns has nan: {}'.format(torch.sum(torch.isnan(torch.tensor(discounted_reward).float()))))
        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.actor_optimizer.zero_grad()
        for _ in range(actor_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            weight = torch.tensor(np.minimum(np.exp(adv[sample_idx] / beta), max_weight)).float().reshape(-1, 1)
            cur_policy = self.model.actor(torch.FloatTensor(s_batch[sample_idx]))

            if self.continuous_agent:
                probs = -cur_policy.log_probs(torch.tensor(action_batch[sample_idx]).float())
                actor_loss = probs * weight
            else:
                m = Categorical(F.softmax(cur_policy, dim=-1))
                actor_loss = -m.log_prob(torch.LongTensor(action_batch[sample_idx])) * weight.reshape(-1)

            actor_loss = actor_loss.mean()
            # print(actor_loss)

            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

        #print('Weight has nan {}'.format(torch.sum(torch.isnan(weight))))

    def save(self, path):
        torch.save(self.model, path)
    def load(self, path):
        self.model = torch.load(path)

def discount_return(reward, done, value):
    value = value.squeeze()
    num_step = len(value)
    discounted_return = np.zeros([num_step])

    gae = 0
    for t in range(num_step - 1, -1, -1):
        if done[t]:
            delta = reward[t] - value[t]
        else:
            delta = reward[t] + gamma * value[t + 1] - value[t]
        gae = delta + gamma * lam * (1 - done[t]) * gae

        discounted_return[t] = gae + value[t]

    # For Actor
    adv = discounted_return - value
    return discounted_return, adv


if __name__ == '__main__':
    agent = ActorAgent(
        state_size,
        action_size,
        gamma,
        use_gae=use_gae,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net,
        use_continuous=continuous)

    states, actions, rewards, next_states, dones = deque(maxlen=max_replay), deque(maxlen=max_replay), deque(
        maxlen=max_replay), deque(maxlen=max_replay), deque(maxlen=max_replay)

    running_reward = []
    success_cnt = 0

    done = False
    episode = 0
    env_info = env.reset(train_mode=train_mode)[default_brain]
    state = env_info.vector_observations[0]

    agent.load(model_path)
    while True:
        action = agent.get_action(state)
        env_info = env.step(action)[default_brain]
        state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        running_reward.append(reward)

        if done:
            episode += 1
            env_info = env.reset(train_mode=train_mode)[default_brain]
            state = env_info.vector_observations[0]

            success = reward == 1
            print('episode: {}, reward : {}, success : {}'.format(episode, np.sum(running_reward), success))
            running_reward = []



