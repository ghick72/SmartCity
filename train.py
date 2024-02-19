import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
from collections import deque
from env_10_10 import SmartCityEnvironment 
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

HEIGHT = 5
WIDTH = 5
UNIT = 100

class SmartCityGym(gym.Env):
    def __init__(self, window=None):
        self.window = window
        if window is not None:
            self.window.title("Smart City Simulation")
            self.canvas = tk.Canvas(window, bg='white', height=500, width=500)
            self.draw_grid()
            self.canvas.pack()
        else:
            self.canvas = None  # GUI 없이도 동작하도록 None으로 설정
            
        super(SmartCityGym, self).__init__()
        self.env = SmartCityEnvironment(None)  # SmartCityEnvironment 인스턴스 생성

        # 상태 공간 및 행동 공간 정의
        self.action_space = spaces.Discrete(len(self.env.actions))  # 가능한 행동의 수
        # 상태는 금액, 인구, 이탈률 등의 연속적인 값들로 구성되며, 각 건물 상태를 포함합니다.
        # 여기서는 상태 공간의 크기를 예제로 설정하였으나, 실제 환경에 맞게 조정해야 합니다.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

    def draw_grid(self):
        if self.canvas is not None:
            for c in range(0, WIDTH * UNIT, UNIT):
                self.canvas.create_line(c, 0, c, HEIGHT * UNIT)
            for r in range(0, HEIGHT * UNIT, UNIT):
                self.canvas.create_line(0, r, WIDTH * UNIT, r)
                
    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
    
class DQN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 할인율
        self.epsilon = 1.0  # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episode_count):
    env = SmartCityGym()
    state_size = 34
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):  # 한 에피소드 당 최대 시간 설정
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if len(agent.memory) > 32:
            agent.replay(32)
        print(f"Episode {e+1}/{episode_count}, epsilon: {agent.epsilon}")

train_dqn(100)  # 에피소드 수 설정
