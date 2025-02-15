import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque, OrderedDict


# Defining the DDQN agent
class DDQN:
    def __init__(self, state_keys, num_actions, model, target_model, learning_rate=0.001,
                 discount_factor=0.95, batch_size=4, memory_size=150000, device=torch.device("cuda")):
        self.state_keys = state_keys
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()
        self.device = device

    def preprocess_state(self, state):
        """
        Preprocesses the state into a flattened numpy array.
        Handles both dictionary and numpy array inputs.
        :param state: Dictionary or numpy array from the environment observation.
        :return: Flattened numpy array representing the state.
        """

        if isinstance(state, dict):  # Handle dictionary state
            features = []
            for key in self.state_keys:
                value = state.get(key)
                if value is not None:
                    if isinstance(value, np.ndarray):
                        features.extend(value.flatten())
                    elif isinstance(value, (float, int)):
                        features.append(value)
                    else:
                        raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")
                else:
                    raise KeyError(f"Key '{key}' not found in the observation dictionary.")
            return torch.tensor(features, dtype=np.float32, device=self.device)
        elif isinstance(state, np.ndarray):
            return torch.tensor(state.flatten(), dtype=torch.float32, device=self.device)
        elif isinstance(state, torch.Tensor):
            return state.to(self.device).flatten()
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((self.preprocess_state(state), action, reward, self.preprocess_state(next_state), done))

    def update_target_model(self):
        """
        Synchronizes the target model with the main model.
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        """
        Returns the best action following epsilon-greedy policy for the current state.
        :param state: Current state.
        :param epsilon: Exploration rate.
        :return: Selected action index.
        """

        state = self.preprocess_state(state).unsqueeze(0).to(self.device)

        if np.random.random() < epsilon:
            # Exploration: Random action in continuous space
            motor = np.random.uniform(-1.0, 1.0)
            steering = np.random.uniform(-1.0, 1.0)
        else:
            # Exploitation: Use model to predict motor and steering values
            with torch.no_grad():
                q_values = self.model(state)
            motor = np.clip(q_values[0][0].item(), -1.0, 1.0)
            steering = np.clip(q_values[0][1].item(), -1.0, 1.0)

        return OrderedDict(
            [('motor', np.array([motor], dtype=np.float32)), ('steering', np.array([steering], dtype=np.float32))])

    def load(self, model_name, episode):
        """
        Loads the weights of the model at the specified episode checkpoint.
        :param model_name: Name of the model.
        :param episode: Episode checkpoint.
        """

        folder_name = os.path.join('..', 'agents')
        os.makedirs(folder_name, exist_ok=True)
        self.model.load_state_dict(torch.load(f'{folder_name}/trained/{model_name}_{episode}.pth', weights_only=True))

    def save(self, model_name, episode):
        """
        Stores the weights of the model at the specified episode checkpoint.
        :param model_name: Name of the model.
        :param episode: Episode checkpoint.
        """

        folder_name = os.path.join('..', 'agents')
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.model.state_dict(), f'{folder_name}/trained/{model_name}_{episode}.pth')

    def train(self):
        """
        Performs one step of model training using experience replay.
        """

        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states, target_q_values = [], []

        for state, action, reward, next_state, done in minibatch:
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            with torch.no_grad():
                next_state_q_values = self.model(next_state.unsqueeze(0))
                best_motor_action_idx = next_state_q_values[0][0].argmax().item()  # Best motor action index
                best_steering_action_idx = next_state_q_values[0][1].argmax().item()  # Best steering action index
                future_q_values = self.target_model(next_state.unsqueeze(0))

            motor_q_value = (reward + self.discount_factor *
                             future_q_values[0, best_motor_action_idx].item() * (1 - done))
            steering_q_value = (reward + self.discount_factor *
                                future_q_values[0, best_steering_action_idx].item() * (1 - done))

            current_q_values = self.model(state.unsqueeze(0))
            current_q_values[0, 0], current_q_values[0, 1] = motor_q_value, steering_q_value
            states.append(state)
            target_q_values.append(current_q_values.squeeze(0))

        states = torch.stack(states)
        target_q_values = torch.stack(target_q_values)
        scaler = torch.amp.GradScaler('cuda')

        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):  # Enabling mixed precision
            output = self.model(states)
            loss = self.criterion(output, target_q_values)

        # Scales the loss and performs the backward pass
        scaler.scale(loss).backward()
        # Unscales the gradients and updates the model parameters
        scaler.step(self.optimizer)
        # Updates the scale for next iteration
        scaler.update()
