from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib

from SC_0401 import SmartCityEnvironment

import pandas as pd
import os
import sys
from tensorflow.keras.initializers import RandomUniform


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 20000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=40000)

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(lr=self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  ## step마다 입실론 감소

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 학습 파라메터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
    
    def save_model(self, filename):
        self.model.save(filename)

EPISODES = 3000

# 환경 및 에이전트 초기화
env = SmartCityEnvironment()
state_size = 11
action_size = 4
agent = DQNAgent(state_size, action_size)

episode_rewards = []
scores = []
episodes = []

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(1040):  # 520주(10년)
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.append_sample(state, action, reward, next_state, done)

        if len(agent.memory) >= agent.train_start:
                agent.train_model()

        state = next_state

        if done == True:
            # print(f"에피소드: {e+1}/{EPISODES}, 총 보상: {total_reward}")

            # if agent.check_early_stop(total_reward, e):
            #     break

            with open(f"./save_model/episode_results{current_time}.txt", "a") as file:
                file.write(f"에피소드={e+1},아파트수={env.num_apartment}, 병원수={env.num_hospitals}, 기지국수={env.num_base_station}, ITS수={env.num_ITS}, 수용 가능 인원={env.capacity_population}, 인구수={env.population}, 누적자본={env.capital}, 누적 만족도={env.happiness}, 누적보상={total_reward}\n")

            # print(f"스탭 {time}: 아파트수={env.apartments}, 병원수={env.hospitals}, 식당수={env.restaurants}, 인구수={env.population}, 누적자본={env.capital}, 누적보상={total_reward}")

            if e+1 == EPISODES:
                agent.save_model(f"./save_model/model_save{current_time}.h5")  # 모델 저장
            
            # 에피소드마다 학습 결과 및 그래프 업데이트
            print("Episode: {:0f} | Reward: {:0f} | Epsilon: {:.3f}".format(e + 1, total_reward, agent.epsilon))
            scores.append(total_reward)
            episodes.append(e + 1)
            
            plt.plot(episodes, scores, 'b')
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.title(f"City_Reward")
            plt.savefig(f"graph_trained.png")
            if EPISODES == e+1:
                plt.savefig(f"./save_model/graph_trained_{current_time}.png")
            plt.close()

            # if agent.check_early_stop(total_reward, e):
            #     break

            break

    episode_rewards.append(total_reward)

file_name = 'Same.xlsx'

# scores 리스트를 데이터프레임으로 변환
new_data = pd.DataFrame(scores)
if os.path.isfile(file_name):
    # 파일이 존재할 경우, 기존 데이터를 불러온 후 새 데이터를 추가
    old_data = pd.read_excel(file_name, header=None, engine='openpyxl')
    # 새로운 데이터를 기존 데이터에 추가
    updated_data = pd.concat([old_data, new_data], ignore_index=True, axis=1)
else:
    # 파일이 존재하지 않을 경우, 새 데이터프레임을 사용
    updated_data = new_data

# 업데이트된 데이터프레임을 엑셀 파일로 저장 (열 이름 없이)
updated_data.to_excel(file_name, index=False, header=False)

# 엑셀파일(이름 : scores) 같은 폴더에 넣어두고 3번이든 4번이든 실행