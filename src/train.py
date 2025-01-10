from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import random
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# This version is complete rewritten as the neural network based version did not work at all. (Stuck at 1e10)
# A lot of ideas did not work. Some parts are remnant of the previous versions, such as the EMA and the Gumbel top 2.
# The maximum reward is probably around 4.5e10, which is the absolute best I could get.
# Indeed, theoretically, to achieve 5e10 in 200 steps, we would need 140 steps on maximum reward, which means that observation[5] has to reach the maximum value at turn 60.
# Training was done on a separate Jupyter notebook

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ExtraTreesRegressor is faster both in training and inference, but the model is too large (400 Mb)
# GradientBoostingRegressor is slower in training and inference, but the model is much smaller (2 Mb)
# A training script is provided in the Jupyter notebook. You can run it to train a new model. 10 epochs are enough to pass 7 tests.

# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
        # Store the exponential moving average of the rewards
        self.alpha = 0.99999
        self.EMA = 5e6
        self.action_set = [
            np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]
        ]
        self.reward_list = []
    def append(self, s, a, r, s_, d, total_reward):
        if len(self.data) < self.capacity:
            self.data.append(None)
        # observation + action
        cls_input = np.concatenate([s, self.action_set[a]])
        self.data[self.index] = (s, a, r, s_, d, total_reward, cls_input)
        self.index = (self.index + 1) % self.capacity
        # Update the EMA
        self.EMA = self.alpha * self.EMA + (1 - self.alpha) * total_reward
        self.reward_list.append(total_reward)
        #self.EMA_record.append(self.EMA)
    def compute_quantile_75(self):
        return np.quantile(self.reward_list, 0.75)
    def compute_median(self):
        return np.median(self.reward_list)
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:np.array(np.array(x)), list(zip(*batch))))
    def get_full_data(self):
        return list(map(lambda x:np.array(np.array(x)), list(zip(*self.data))))
    def __len__(self):
        return len(self.data)
    def reset(self):
        self.data = []
        self.index = 0
        self.reward_list = []
 
def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(U + eps) + eps)


class ProjectAgent:

    def __init__(self):
        #self.classifier = GradientBoostingClassifier()
        #self.classifier = ExtraTreesClassifier(n_estimators=50)
        #self.regressor = ExtraTreesRegressor(n_estimators=50)
        #self.regressor = GradientBoostingRegressor(n_estimators=50)
        self.regressor = HistGradientBoostingRegressor()
        self.action_set = [
            np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]
        ]
        self.memory = ReplayBuffer(60000, 'cpu')
        self.gamma = 0.98

    def q_values(self, observation):
        q = []
        for action in self.action_set:
            # Append the action to the observation
            cls_input = np.concatenate([observation, action]).reshape(1, -1)
            #q.append(self.classifier.predict_proba(cls_input)[0][1])
            q.append(self.regressor.predict(cls_input)[0])
        return np.array(q)

    def epsilon_greedy_action(self, observation, epsilon):
        if random.random() < epsilon:
            return np.random.randint(4)
        else:
            Q = self.q_values(observation)
            return np.argmax(Q)

    """
    def gumbel_top_2(self, observation):
        gumbel_values = sample_gumbel(4)
        proba = self.classifier.predict_proba(observation.reshape(1, -1))[0]
        logits = np.log(proba + 1e-20)
        top_2 = np.argsort(gumbel_values + logits)[-2:]

        # Find the highest Q among the top 2
        Q = self.q_values(observation)
        return top_2[np.argmax(Q[top_2])]
    """


    def complete_one_episode(self, env, epsilon, pure_random=False):
        observation, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if pure_random:
                action = np.random.randint(4)
            else:
                action = self.epsilon_greedy_action(observation, epsilon)
                #action = self.gumbel_top_2(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward
            if not done:
                self.memory.append(observation, action, reward, next_observation, done, total_reward)
            observation = next_observation

    def train_on_data(self, first_iteration=False):
        """
        #threshold = self.memory.compute_quantile_75()
        threshold = self.memory.compute_median()
        print(threshold)
        s, a, r, s_, d, total_reward, cls_input = self.memory.get_full_data()
        # Train test split
        X = cls_input
        # Set classes
        y = total_reward > threshold
        y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
        # Train the model
        self.classifier.fit(X_train, y_train)
        # Evaluate the model
        print(self.classifier.score(X_test, y_test))
        """
        s, a, r, s_, d, total_reward, cls_input = self.memory.get_full_data()
        X = cls_input

        # Q-learning
        next_y = np.zeros((X.shape[0], 4))
        if not first_iteration:
            for action in range(4):
                next_state = np.concatenate([s_, np.tile(self.action_set[action], (s_.shape[0], 1))], axis=1)
                next_y[:, action] = self.regressor.predict(next_state)
            y = r + self.gamma * np.max(next_y, axis=1)
        else:
            y = r

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Train the model
        #self.regressor.fit(X_train, y_train)
        # Evaluate the model
        #print(self.regressor.score(X_test, y_test))
        self.regressor.fit(X, y)
        #self.classifier.fit(s, a)

    def act(self, observation, use_random=False):
        Q = self.q_values(observation)
        #P = self.classifier.predict_proba(observation.reshape(1, -1))[0]
        return np.argmax(Q)

    def save(self, path):
        # Save the model
        joblib.dump(self.regressor, "hist_gradient_boosting_regressor.joblib")
        #joblib.dump(self.classifier, "extra_trees_classifier.joblib")

    def load(self):
        self.regressor = joblib.load("hist_gradient_boosting_regressor.joblib")
        #self.classifier = joblib.load("extra_trees_classifier.joblib")
