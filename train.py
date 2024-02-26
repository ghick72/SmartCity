import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pylab
import os
from a import SmartCityEnvironment

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 환경 및 에이전트 초기화
state_size = 34  # 상태의 크기를 정의 (예시로 45를 사용, 상황에 맞게 조정)
action_size = 5 * 5 * 5  # 가능한 행동의 수 (5x5 그리드에 5개의 건물 유형)
agent = DQNAgent(state_size, action_size)
scores, episodes = [], []  # 에피소드별 점수를 저장하기 위한 리스트

if not os.path.exists("save_graph"):
    os.makedirs("save_graph")
    
env = SmartCityEnvironment()
EPISODES = 500
# DQN 학습 시작
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        score += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print(f"Episode: {e+1}, score: {score}, epsilon: {agent.epsilon:.2f}")
            scores.append(score)
            episodes.append(e)
    batch_size = 32
    # 매 에피소드마다 agent.replay()를 호출하여 경험 리플레이
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    env.mainloop()

    # 모든 에피소드가 종료된 후, 그래프 저장
    if (e + 1) % 10 == 0:  # 매 10회의 에피소드마다 그래프 업데이트
        pylab.figure(figsize=(10, 5))
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("Episode")
        pylab.ylabel("Score")
        pylab.savefig("./save_graph/smart_city_dqn.png")
