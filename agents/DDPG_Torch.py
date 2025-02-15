import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, OrderedDict
import os
from agents.QNetwork import QNetwork


# Defining the DDPG agent
class DDPG:
    def __init__(self, state_keys, state_dim, action_dim, actor_hidden_sizes=(128, 64), critic_hidden_sizes=(128, 64),
                 learning_rate=0.001, discount_factor=0.95, batch_size=4, memory_size=150000,
                 device=torch.device("cpu")):

        self.state_keys = state_keys
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Initializing actor and critic models
        self.actor = QNetwork(state_dim, action_dim, actor_hidden_sizes).to(device)
        self.critic = QNetwork(state_dim + action_dim, 1, critic_hidden_sizes).to(device)

        # Initializing target actor and critic models
        self.target_actor = QNetwork(state_dim, action_dim, actor_hidden_sizes).to(device)
        self.target_critic = QNetwork(state_dim + action_dim, 1, critic_hidden_sizes).to(device)

        # Synchronize target models with main models
        self.update_target_model()

        # Initializing optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.device = device

    def preprocess_state(self, state):
        """
        Preprocesses the state into a tensor.
        :param state: Dictionary or numpy array from the environment observation.
        :return: Tensor representing the state.
        """

        if isinstance(state, dict):
            features = []
            for key in self.state_keys:
                value = state.get(key)
                if value is not None:
                    if isinstance(value, np.ndarray):
                        features.append(value.flatten())
                    elif isinstance(value, (float, int)):
                        features.append(value)
                    else:
                        raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")
                else:
                    raise KeyError(f"Key '{key}' not found in the observation dictionary.")

            return torch.tensor(features, dtype=torch.float32, device=self.device)
        elif isinstance(state, np.ndarray):  # Handle ndarray state
            return torch.tensor(state.flatten(), dtype=torch.float32, device=self.device)
        elif isinstance(state, torch.Tensor):
            return state.to(self.device).flatten()
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((self.preprocess_state(state), action, reward, self.preprocess_state(next_state), done))

    def update_target_model(self):
        """
        Updates the target models with the main models.
        """

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def get_action(self, state, epsilon):
        """
        Returns the action predicted by the actor network.
        :param state: Current state.
        :param epsilon: Exploration rate.
        :return: Selected action (motor and steering values).
        """

        state = self.preprocess_state(state).unsqueeze(0).to(self.device)

        if np.random.random() < epsilon:
            # Exploration: Random action in continuous space
            motor = np.random.uniform(-1.0, 1.0)
            steering = np.random.uniform(-1.0, 1.0)
        else:
            # Exploitation: Use actor model to predict motor and steering values
            with torch.no_grad():
                action = self.actor(state)
                motor, steering = action[0]  # Get the motor and steering from the actor

        # Ensuring actions are on CPU and clip within the valid range
        motor = motor.cpu().numpy() if isinstance(motor, torch.Tensor) else motor
        steering = steering.cpu().numpy() if isinstance(steering, torch.Tensor) else steering

        # Clipping to the valid range based on the car's actuator limits
        motor = np.clip(motor, -1.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)

        return OrderedDict(
            [('motor', np.array([motor], dtype=np.float32)), ('steering', np.array([steering], dtype=np.float32))]
        )

    def load(self, model_name, episode):
        """
        Loads the actor and critic models at the specified episode checkpoint.
        """

        folder_name = os.path.join('..', 'agents')
        os.makedirs(folder_name, exist_ok=True)
        self.actor.load_state_dict(torch.load(f'{folder_name}/trained/{model_name}_actor_{episode}.pth',
                                              weights_only=True))
        self.critic.load_state_dict(torch.load(f'{folder_name}/trained/{model_name}_critic_{episode}.pth',
                                               weights_only=True))

    def save(self, model_name, episode):
        """
        Saves the actor and critic models at the specified episode checkpoint.
        """

        folder_name = os.path.join('..', 'agents')
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{folder_name}/trained/{model_name}_actor_{episode}.pth')
        torch.save(self.critic.state_dict(), f'{folder_name}/trained/{model_name}_critic_{episode}.pth')

    def train(self):
        """
        Performs one step of model training using experience replay.
        """

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for state, action, reward, next_state, done in minibatch:
            states.append(state)

            # Convert OrderedDict (action) to tensors
            motor = torch.tensor(action['motor'], dtype=torch.float32).unsqueeze(0)
            steering = torch.tensor(action['steering'], dtype=torch.float32).unsqueeze(0)
            actions.append(torch.cat([motor, steering], dim=-1))

            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Converting lists to tensors and moving to device
        states = torch.stack(states).to(self.device).to(torch.float16)
        actions = torch.stack(actions).to(self.device).to(torch.float16)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        actions = actions.view(-1, self.action_dim)  # Flatten to (batch_size, action_dim)
        states = states.view(-1, self.state_dim)  # Flatten to (batch_size, state_dim)

        scaler = torch.amp.GradScaler(enabled=True)

        # Updating the critic
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # Enable mixed precision
            with torch.no_grad():
                target_actions = self.target_actor(next_states)
                target_q_values = self.target_critic(torch.cat([next_states, target_actions], dim=-1))
                target_values = rewards + self.discount_factor * target_q_values * (1 - dones)
                target_values = target_values.to(torch.float16)

        with torch.autocast(device_type="cuda", dtype=torch.float16):  # Enable mixed precision
            q_values = self.critic(torch.cat([states.to(torch.float16), actions], dim=-1))

        # Debugging dtype of tensors before loss calculation
        # print(f"states dtype: {states.dtype}")
        # print(f"actions dtype: {actions.dtype}")
        # print(f"q_values dtype: {q_values.dtype}")
        # print(f"target_values dtype: {target_values.dtype}")

        # Reshaping target and current Q-values to be compatible for loss calculation
        target_values = target_values.view(-1, 1)
        q_values = q_values.view(-1, 1)

        critic_loss = nn.MSELoss()(q_values, target_values)  # Mean Squared Error loss for critic

        self.critic_optimizer.zero_grad()
        scaler.scale(critic_loss).backward()  # Scaling the loss for mixed precision
        scaler.step(self.critic_optimizer)  # Stepping with scaled gradients
        scaler.update()  # Updating the scaler for next iteration

        # Updating the actor
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # Enable mixed precision
            actor_loss = -self.critic(torch.cat([states.to(torch.float16), self.actor(states.to(torch.float16))],
                                                dim=-1)).mean()  # Maximize Q-value

        # Backpropagation with mixed precision for actor
        self.actor_optimizer.zero_grad()
        scaler.scale(actor_loss).backward()  # Scaling the loss for mixed precision
        scaler.step(self.actor_optimizer)  # Stepping with scaled gradients
        scaler.update()  # Updating the scaler

        self.update_target_model()
