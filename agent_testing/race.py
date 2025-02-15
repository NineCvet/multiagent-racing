import torch
import gymnasium
import numpy as np
from agents.DDPG_Torch import DDPG
from agents.DDQN_Torch import DDQN
from agents.DQN_Torch import DQN
from agents.QNetwork import QNetwork
from racecar_gym.envs import pettingzoo_api
from data_analysis.save_csv import save_episode_to_csv
import time
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_agent(agent_id, state_keys, num_actions, state_size, device):
    if agent_id == "A":
        # Initializing DDPG agent
        actor_hidden_sizes = [256, 128]
        critic_hidden_sizes = [256, 128]
        agent = DDPG(
            state_keys=state_keys,
            state_dim=state_size,
            action_dim=num_actions,
            learning_rate=0.001,
            discount_factor=0.95,
            tau=0.005,
            batch_size=64,
            memory_size=100000,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            device=device
        )
    elif agent_id == "B":
        # Initializing DDQN agent
        model = QNetwork(state_size, num_actions).to(device)
        target_model = QNetwork(state_size, num_actions).to(device)
        agent = DDQN(
            state_keys=state_keys,
            num_actions=num_actions,
            model=model,
            target_model=target_model,
            device=device
        )
    elif agent_id == "C":
        # Initializing DQN agent
        model = QNetwork(state_size, num_actions).to(device)
        target_model = QNetwork(state_size, num_actions).to(device)
        agent = DQN(
            state_keys=state_keys,
            num_actions=num_actions,
            model=model,
            target_model=target_model,
            device=device
        )
    else:
        raise ValueError("Unknown agent id: " + agent_id)
    return agent


def race():
    scenario_path = '../scenarios/circle_race.yml'

    # Creating the environment using pettingzoo_api
    env = pettingzoo_api.env(
        scenario=scenario_path,
        render_mode='rgb_array_birds_eye',
        render_options=dict(width=320, height=240)
    )

    # print(env.reset(return_info=True))

    # for agent_id in env.agents:
    #     print(f"Observation space for agent {agent_id}: {env.observation_space(agent_id)}")

    state_keys = ['pose', 'velocity', 'acceleration', 'time']
    state_size = 0

    agent_id = 'A'
    obs_space = env.observation_space(agent_id)

    for key in state_keys:
        space = obs_space[key]
        print(f"Key: {key}, Shape: {space.shape}")
        if isinstance(space, gymnasium.spaces.Box):
            if len(space.shape) == 0:
                # Scalar value for feature
                print(f"  Scalar value for {key}")
                state_size += 1
            else:
                # Feature size is the first dimension of the shape
                feature_size = space.shape[0]
                print(f"  Feature size for {key}: {feature_size}")
                state_size += feature_size

    # print(f"Calculated state_size: {state_size}")  # Debugging line
    if state_size == 0:
        raise ValueError("The calculated state_size is zero!")

    num_actions = 2

    # Initializing agents A, B, and C
    agents = {
        "A": create_agent("A", state_keys, num_actions, state_size, device),
        "B": create_agent("B", state_keys, num_actions, state_size, device),
        "C": create_agent("C", state_keys, num_actions, state_size, device)
    }

    # Training parameters
    episodes = 1
    epsilon = 0.01
    render = False
    rank_csv_file = 'agents_check.csv'
    max_steps = 10000

    agents['A'].load('racecar_ddpg_only_multi', 1000)
    agents['B'].load('racecar_ddqn_only_multi', 1000)
    agents['C'].load('racecar_dqn_only_multi', 1000)

    # Training loop
    for episode in range(episodes):
        start_time = time.time()
        obs, info = env.reset(return_info=True)

        # Initializing states for all agents
        states = {}
        for agent_id in obs.keys():
            states[agent_id] = np.concatenate([obs[agent_id][key].flatten() for key in state_keys])
            states[agent_id] = torch.tensor(states[agent_id], dtype=torch.float32, device=device)

        done = False
        total_reward = 0
        episode_ranks = {}
        steps = 0

        while not done or steps < max_steps:
            actions = {}
            dones = {}  # Track done status for each agent
            for agent_id, state in states.items():
                actions[agent_id] = agents[agent_id].get_action(state, epsilon)

            action_dict = {agent_id: actions[agent_id] for agent_id in actions}
            obs, reward, terminated, truncated, info = env.step(action_dict)

            for agent_id in obs.keys():

                dones[agent_id] = truncated or terminated

            done = all(dones.values())
            #print(info)

            next_states = {}
            for agent_id in obs.keys():
                next_state = np.concatenate([
                    obs[agent_id]['pose'].flatten(),
                    obs[agent_id]['velocity'].flatten(),
                    obs[agent_id]['acceleration'].flatten(),
                    obs[agent_id]['time'].flatten()
                ])
                # print(f"Next state shape: {next_state.shape}")
                next_states[agent_id] = torch.tensor(next_state, dtype=torch.float32, device=device)

            # Update states for next iteration
            states = next_states
            total_reward += sum(reward.values())

            steps += 1
            # Rendering the environment if needed
            if render:
                image = env.render()
                image = np.array(image, dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Rendered Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        agent_ranks = {agent_id: info[agent_id]['rank'] for agent_id in info.keys()}
        episode_ranks[episode + 1] = agent_ranks

        # Logging episode performance
        end_time = time.time()
        episode_time = end_time - start_time  # Time taken for the episode
        print(
            f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward:.2f} "
            f"- Epsilon: {epsilon:.3f} - Time Taken: {episode_time:.2f} seconds")

        processed_data = {
            'Race': episode + 1,
            'Agent A Rank': agent_ranks.get('A', 'N/A'),
            'Agent B Rank': agent_ranks.get('B', 'N/A'),
            'Agent C Rank': agent_ranks.get('C', 'N/A')
        }

        save_episode_to_csv(rank_csv_file, processed_data)

    # Closing the environment
    env.close()
    cv2.destroyAllWindows()


# Race Start
race()
