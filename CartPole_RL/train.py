import gym
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Load environment
env = gym.make("CartPole-v1")
nb_actions = env.action_space.n

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

# Configure the agent
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=nb_actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(Adam(learning_rate=0.001), metrics=["mae"])

# Train the agent
agent.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Test the agent
results = agent.test(env, nb_episodes=5, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()