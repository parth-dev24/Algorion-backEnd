import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense
import gym
from gym import spaces
import random

class StockEnv(gym.Env):
    def __init__(self, df):
        super(StockEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(60, 1), dtype=np.float32)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step:self.current_step + 60]

    def step(self, action):
        self.current_step += 1
        
        # Check if done before calculating next state and reward
        done = self.current_step >= len(self.data) - 60
        
        if done:
            # If done, return the current state, 0 reward, and done flag
            return self.data[self.current_step-1:self.current_step + 59], 0, True, {}
        
        # If not done, calculate next state and reward as before
        next_state = self.data[self.current_step:self.current_step + 60]
        if action == 1:  # buy
            reward = self.data[self.current_step + 60 - 1] - self.data[self.current_step + 59 - 1]  # Adjust indices for 0-based indexing
        elif action == 2:  # sell
            reward = self.data[self.current_step + 59 - 1] - self.data[self.current_step + 60 - 1]  # Adjust indices for 0-based indexing
        else:
            reward = 0  # Hold action, no reward
            
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32)

    def predict(self, X):
        return self.model.predict(X)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.state_size, 1)))
        model.add(LSTM(50))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
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
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Fetch stock data
def fetch_stock_data(symbol, period='2y'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    return df

# Prepare data
def prepare_data(df, sequence_length=60):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test, scaler

# Main function
def main():
    symbol = "RELIANCE.NS"
    period = "2y"
    df = fetch_stock_data(symbol, period)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Initialize and train LSTM model
    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.train(X_train, y_train, epochs=10)

    # Initialize environment
    env = StockEnv(df)

    # Initialize agent
    state_size = 60
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    batch_size = 32

    # Train the agent
    for e in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size, 1])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size, 1])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{1000}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

if __name__ == "__main__":
    main()