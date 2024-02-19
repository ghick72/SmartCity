import tkinter as tk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import threading
from queue import Queue
import threading
import time
from env_10_10 import SmartCityEnvironment

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def run():
    env = SmartCityEnvironment(root)  # 환경 인스턴스 생성
    # 기타 상태 정보의 크기
    base_state_size = 9  # 자본, 인구, 이탈률, 주거공간 개수, 상업공간 개수, 산업공간 개수, 병원 개수, 공원 개수, 행복도

    # 그리드 내 모든 위치에 대한 건물 상태 정보의 크기
    buildings_state_size = 25  # 여기서는 각 위치가 건물 유형을 직접 나타내므로 추가 인코딩 없이 1차원으로 표현 가능

    # 최종 state_size
    state_size = base_state_size + buildings_state_size
    action_size = 125
    agent = DQNAgent(state_size, action_size)
    episodes = 30000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(1000):  # 또는 환경의 최대 스텝 수
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
        if len(agent.memory) > 32:
            agent.replay(32)

# 백그라운드 작업을 위한 함수
def background_task(queue):
    for i in range(10):
        # 처리 결과를 queue에 추가
        queue.put(f"작업 {i} 완료")
        time.sleep(1)  # 시뮬레이션을 위해 1초 대기

# GUI를 업데이트하는 함수
def update_gui(queue):
    try:
        # queue에서 결과를 가져와서 GUI 업데이트
        result = queue.get_nowait()
        print(result)  # 예시로 콘솔에 출력, 실제로는 GUI 업데이트 코드
    except:
        pass
    root.after(100, update_gui, queue)  # 100ms 후에 다시 update_gui 함수를 호출

root = tk.Tk()
queue = Queue()

# 백그라운드 스레드 시작
threading.Thread(target=background_task, args=(queue,)).start()

# 메인 이벤트 루프에서 GUI 업데이트 스케줄링
root.after(100, update_gui, queue)

root.mainloop()


        
# root = tk.Tk()

# env = SmartCityEnvironment(root)  # 환경 인스턴스 생성
# # 기타 상태 정보의 크기
# base_state_size = 9  # 자본, 인구, 이탈률, 주거공간 개수, 상업공간 개수, 산업공간 개수, 병원 개수, 공원 개수, 행복도

# # 그리드 내 모든 위치에 대한 건물 상태 정보의 크기
# buildings_state_size = 25  # 여기서는 각 위치가 건물 유형을 직접 나타내므로 추가 인코딩 없이 1차원으로 표현 가능

# # 최종 state_size
# state_size = base_state_size + buildings_state_size
# action_size = 125
# agent = DQNAgent(state_size, action_size)
# episodes = 30000

# for e in range(episodes):
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])

#     for time in range(1000):  # 또는 환경의 최대 스텝 수
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
#             break
#     if len(agent.memory) > 32:
#         agent.replay(32)
        
# root.mainloop()