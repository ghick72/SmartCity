from a import SmartCityEnvironment 
import pylab
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from shapely.geometry import Polygon, Point
import time

# 환경 설정
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로
ACTION_SIZE = 5 * 5 * 5  # (x, y) 위치와 건물 유형 (5x5 그리드, 5종류의 건물)


# DQN 모델
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.fc_out = Dense(action_size, activation='linear')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc_out(x)

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = DQN(action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.target_model = DQN(action_size)
        self.update_target_model()
        self.memory = deque(maxlen=2000)  # 경험 재생 메모리 초기화

    def append_sample(self, state, action, reward, next_state, done):
        """경험 재생 메모리에 샘플을 추가합니다."""
        self.memory.append((state, action, reward, next_state, done))
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            q_values = self.model(state)
            return np.argmax(q_values[0])

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

if __name__ == "__main__":
    env = SmartCityEnvironment(render_speed=0.01)
    state_size = 34  # 예시 상태 벡터 크기
    action_size = ACTION_SIZE
    agent = DQNAgent(state_size, action_size)

    EPISODES = 500
    scores = []  # 에피소드별 점수 저장
    episodes = []  # 에피소드 번호 저장
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            score += reward

            agent.append_sample(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.batch_size:
                agent.train_model()

            state = next_state

            if done:
                agent.update_target_model()
                print(f"Episode: {e+1}, score: {score}, epsilon: {agent.epsilon:.2f}")
                # 에피소드마다 학습 결과 저장
                scores.append(score)
                episodes.append(e)

        env.mainloop()
        
        # 모든 에피소드가 종료된 후, 그래프 저장
        if (e + 1) % 10 == 0:  # 매 10회의 에피소드마다 그래프 업데이트
            pylab.figure(figsize=(10, 5))
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("Episode")
            pylab.ylabel("Score")
            pylab.savefig("./save_graph/smart_city_dqn.png")

