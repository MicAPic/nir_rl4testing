from collections import Counter

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.regularizers import l2
from keras.losses import Huber
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import InverseTimeDecay

BEST_SCORE = 190  # DQN


class RetentionReplayBuffer:
    def __init__(
            self,
            capacity,
            input_shape,
            retention_rate: float = 0.1
    ):
        self.capacity = capacity
        self.retention_rate = retention_rate
        self.counter = 0
        self.next_idx = -1
        self.state_buffer = np.zeros((self.capacity, input_shape), dtype=np.float32)
        self.action_buffer = np.zeros(self.capacity, dtype=np.int32)
        self.reward_buffer = np.zeros(self.capacity, dtype=np.float32)
        self.new_state_buffer = np.zeros((self.capacity, input_shape), dtype=np.float32)
        self.terminal_buffer = np.zeros(self.capacity, dtype=np.bool_)

    def add(self, state, action, reward, new_state, done):
        if self.counter >= self.capacity:
            self.next_idx = -1 + int(self.capacity * self.retention_rate)
            self.counter = self.next_idx + 1
            print("\n=== Eviction started here ===\n")
        else:
            self.next_idx += 1
        self.state_buffer[self.next_idx] = state
        self.action_buffer[self.next_idx] = action
        self.reward_buffer[self.next_idx] = reward
        self.new_state_buffer[self.next_idx] = new_state
        self.terminal_buffer[self.next_idx] = done
        self.counter += 1

    def sample(self, batch_size):
        batch = np.random.choice(self.counter, batch_size, replace=False)
        state_batch = self.state_buffer[batch]
        action_batch = self.action_buffer[batch]
        reward_batch = self.reward_buffer[batch]
        new_state_batch = self.new_state_buffer[batch]
        done_batch = self.terminal_buffer[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch


# def dqn(learning_rate, num_actions, input_dims, regularization_factor):
#     model = Sequential([
#         Dense(64, input_dim=input_dims, activation="relu", kernel_regularizer=l2(regularization_factor)),
#         Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
#         Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
#         Dense(num_actions, activation='linear', kernel_regularizer=l2(regularization_factor))
#     ])
#
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss=Huber())
#     return model


class D3QN(keras.Model):
    def __init__(self, num_actions, regularization_factor):
        super(D3QN, self).__init__()  # calls __init__ of the nn.Model class
        self.dense1 = Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor))
        self.dense2 = Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor))
        self.V = Dense(1, activation=None)
        self.A = Dense(num_actions, activation=None)

    def call(self, state, **kwargs):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)  # Advantage
        avg_A = tf.math.reduce_mean(A, axis=1, keepdims=True)
        Q = (V + (A - avg_A))

        return Q, A


class Agent:
    def __init__(self, lr, lr_decay_rate, discount_factor, num_actions, epsilon_decay_rate, batch_size, input_dims):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay_rate
        if lr > 0:
            self.epsilon = 1.0
            self.epsilon_final = 0.01
        else:
            self.epsilon = 0
            self.epsilon_final = 0
        self.update_rate = 1024
        self.iterations = 0
        self.buffer = RetentionReplayBuffer(30_000, input_dims)

        lr_func = InverseTimeDecay(lr, decay_steps=300_000, decay_rate=lr_decay_rate)

        self.q_net = D3QN(num_actions, 0.001)  # main model for selecting actions
        self.q_target_net = D3QN(num_actions, 0.001)  # target model for calculating the target values
        self.q_net.compile(optimizer=Adam(learning_rate=lr_func), loss=Huber())
        self.q_target_net.compile(optimizer=Adam(learning_rate=lr_func), loss=Huber())

    def policy(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            _, actions = self.q_net(state)  # A
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def train(self):
        if self.buffer.counter < self.batch_size:
            return
        if self.iterations % self.update_rate == 0:
            # the weights of the main model are copied to the target model on every update
            self.q_target_net.set_weights(self.q_net.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, terminal_batch = \
            self.buffer.sample(self.batch_size)

        q_predicted, _ = self.q_net(state_batch)
        q_next, _ = self.q_target_net(new_state_batch)
        q_target = np.copy(q_predicted)
        # DQN:
        # q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        #
        # Double DQN:
        _, actions = self.q_net(new_state_batch)  # A
        max_actions = tf.math.argmax(actions, axis=1)

        for idx in range(terminal_batch.shape[0]):
            if not terminal_batch[idx]:
                # DQN:
                # q_target[idx, action_batch[idx]] = reward_batch[idx] + self.discount_factor * q_max_next[idx]
                #
                # Double DQN:
                q_target[idx, action_batch[idx]] = reward_batch[idx] + self.discount_factor * \
                                                   q_next[idx, max_actions[idx]]
            else:
                q_target[idx, action_batch[idx]] = reward_batch[idx]

        self.q_net.train_on_batch(state_batch, q_target)
        self.iterations += 1

    def train_model(self, environment, steps: int, evaluate: bool):

        # global BEST_SCORE
        scores, episodes, avg_scores, bugs_spotted = [], [], [], []
        goal = 200

        for i in range(steps):
            done = False
            score = 0.0
            state, _ = environment.reset()
            bug_spotted = [False, False]
            while not done:
                action = self.policy(state)
                next_state, reward, terminated, truncated, _ = environment.step(action)

                # RELINE
                if -0.9 < next_state[0] < -0.8 and not bug_spotted[0]:
                    reward += 250
                    bug_spotted[0] = True
                if 0.75 < next_state[0] < 0.8 and not bug_spotted[1]:
                    reward += 250
                    bug_spotted[1] = True

                if terminated or truncated:
                    done = True
                    # RELINE
                    if bug_spotted == [False, False]:
                        reward -= 50

                score += reward
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                if not evaluate:
                    self.train()
            scores.append(score)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            bugs_spotted.append(bug_spotted)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_final)

            # use in hyperparameter tuning:
            #
            # if avg_score >= goal:
            #     break
            # elif (avg_score >= BEST_SCORE) and (score >= goal):
            #     self.q_net.save(f"models/{BEST_SCORE}")
            #     BEST_SCORE = avg_score

            if evaluate:
                print(f"Episode {i}, Score: {score}, "
                      # f"bug found: {bug_spotted}, "
                      f"average score: {avg_score}")
            else:
                print(f"Episode {i}, Score: {score} (ε = {self.epsilon}), "
                      f"lr: {self.q_net.optimizer.lr(self.iterations)}, "
                      f"bug found: {bug_spotted}, "
                      f"average score: {avg_score}")

        if not evaluate:
            self.q_net.save(f"models/temp")

        self.plot_results(goal, episodes, scores, avg_scores)
        if evaluate:
            print("Results:")
            [print([a, b], v) for (a, b), v in Counter(map(tuple, bugs_spotted)).items()]

    @staticmethod
    def plot_results(goal, episode_list, score_list, avg_score_list):
        plt.figure()
        plt.xlabel("Счет")
        plt.ylabel("Эпизод")
        plt.plot(episode_list, score_list, marker='', color='blue', linewidth=2)
        plt.plot(episode_list, avg_score_list, marker='', color='orange', linewidth=2, linestyle='dashed',
                 label='Средний счет')
        plt.hlines(goal, xmin=episode_list[0], xmax=episode_list[-1], color='red', linewidth=2, linestyle='dashed',
                   label='Цель')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # training agent
    env = gym.make("LunarLander-v2")
    dqn_agent = Agent(lr=1e-3, lr_decay_rate=10, discount_factor=0.99, num_actions=4, epsilon_decay_rate=0.995,
                      batch_size=32, input_dims=8)
    dqn_agent.train_model(env, steps=250, evaluate=False)
    env.close()

    # evaluating agent
    env = gym.make(
        "LunarLander-v2",
        # render_mode="human"
                   )
    dqn_agent = Agent(lr=0, lr_decay_rate=0, discount_factor=0.99, num_actions=4, epsilon_decay_rate=0,
                      batch_size=32, input_dims=8)
    dqn_agent.q_net = keras.models.load_model("models/temp")
    # dqn_agent.epsilon_final = 1  # uncomment for random agent
    dqn_agent.train_model(env, 100, True)
    env.close()
