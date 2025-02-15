import gymnasium
import numpy as np
import cv2
import os
import torch
from agents.DDPG_Torch import DDPG
from data_analysis.preprocess_data import process_episode_data
from data_analysis.save_csv import save_episode_to_csv
import racecar_gym.envs
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device("cuda")


def train_ddpg():
    scenario_path = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'circle.yml')

    # Creating the environment
    env = gymnasium.make(
        id='SingleAgentRaceEnv-v0',
        scenario=scenario_path,
        render_mode='rgb_array_birds_eye',  # Optional
        render_options=dict(width=320, height=240)  # Optional
    )

    # Extracting environment details
    print("Action space:", env.action_space)
    print("Observation space keys:", env.observation_space.spaces.keys())

    # Defining the state keys and calculating state size
    state_keys = ['pose', 'velocity', 'acceleration', 'time']
    state_size = int(sum(
        env.observation_space[key].shape[0] if isinstance(env.observation_space[key], gymnasium.spaces.Box) and len(env.observation_space[key].shape) > 0
        else 1
        for key in state_keys
    ))
    action_dim = 2

    # Defining hidden sizes for actor and critic networks
    actor_hidden_sizes = [256, 128]
    critic_hidden_sizes = [256, 128]

    # Creating the DDPG agent
    agent = DDPG(
        state_keys=state_keys,
        state_dim=state_size,
        action_dim=action_dim,
        batch_size=4,
        actor_hidden_sizes=actor_hidden_sizes,
        critic_hidden_sizes=critic_hidden_sizes,
        device=device
    )
    # agent.load('racecar_ddpg', 500)

    # Training parameters
    episodes = 1
    epsilon = 1.0  # Initial exploration rate for noise
    epsilon_min = 0.1  # Minimum exploration rate
    epsilon_decay = 0.995  # Decay rate for epsilon
    render = False
    csv_file = 'ddpg_solo_check.csv'
    max_steps = 1000

    # Training loop
    for episode in range(episodes):
        start_time = time.time()
        obs, info = env.reset()
        state = np.concatenate([obs[key].flatten() for key in state_keys])
        state = torch.tensor(state, dtype=torch.float32, device=device)
        done = False
        total_reward = 0
        episode_data = []

        while not done or len(episode_data) < max_steps:
            action = agent.get_action(state, epsilon)

            # Performing action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = np.concatenate([obs[key].flatten() for key in state_keys])
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            done = terminated or truncated

            # Storing experience in memory
            agent.update_memory(state, action, reward, next_state, done)

            agent.train()

            # Logging the step data
            step_data = {
                'time': info['time'],
                'reward': reward,
                'wall_collision': info['wall_collision'],
                'progress': info['progress'],
                'wrong_way': info['wrong_way'],
            }
            episode_data.append(step_data)

            # Updating state
            state = next_state
            total_reward += reward

            # Rendering the environment
            if render:
                image = env.render()
                image = np.array(image, dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Rendered Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Updating target network
        agent.update_target_model()

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Logging episode performance
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward:.2f} "
              f"- Epsilon: {epsilon:.3f} - Time Taken: {episode_time:.2f} seconds")

        processed_data = process_episode_data(episode_data)
        save_episode_to_csv(csv_file, processed_data)

    agent.save('racecar_ddpg', episodes)

    # Closing the environment
    env.close()
    cv2.destroyAllWindows()


# Training
train_ddpg()
