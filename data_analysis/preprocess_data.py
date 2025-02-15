from collections import Counter
import numpy as np


def process_episode_data(episode_data, multi=False):
    """
    Process all steps in an episode to compute averages and statistics.
    :param episode_data: List of dictionaries containing step info (e.g., reward, progress, wrong_way, etc.)
    :param multi: Boolean values to specify if the step data is from multi-agent training or solo training
    :return: Processed episode statistics in a dictionary.
    """
    total_reward = 0
    total_progress = 0
    timed_reward = 0
    timed_progress = 0
    total_time = 0
    wrong_way_count = 0
    collision_count = 0
    num_steps = len(episode_data)
    opponent_collision_count = 0
    rank_counts = Counter()

    reward_10s = []
    progress_10s = []
    time_10s = 10

    for step_data in episode_data:
        total_reward += step_data['reward']
        total_progress += step_data['progress']
        timed_reward += step_data['reward']
        timed_progress += step_data['progress']
        total_time += step_data['time']

        if multi and len(step_data['opponent_collisions']) > 0:
            opponent_collision_count += len(step_data['opponent_collisions'])

        if multi:
            rank_counts[step_data['rank']] += 1

        if step_data['wall_collision'] is True:
            collision_count += 1

        if step_data['wrong_way'] is True:
            wrong_way_count += 1

        # Tracking every 10 seconds for progress and reward
        if total_time >= time_10s:
            reward_10s.append(timed_reward)
            progress_10s.append(timed_progress)
            timed_reward = 0
            timed_progress = 0
            time_10s += 10

    # Calculating average reward and progress over the episode
    avg_reward = total_reward / num_steps if num_steps > 0 else 0
    avg_progress = total_progress / num_steps if num_steps > 0 else 0

    # Calculating average reward and progress every 10 seconds
    avg_reward_10s = np.mean(reward_10s) if reward_10s else 0
    avg_progress_10s = np.mean(progress_10s) if progress_10s else 0

    # Calculating percentage of time going the wrong way
    wrong_way_percentage = (wrong_way_count / num_steps) * 100 if num_steps > 0 else 0
    right_way_percentage = 100 - wrong_way_percentage

    # Determining the most frequent rank in multi-agent training
    if multi:
        most_frequent_rank = rank_counts.most_common(1)[0][0] if rank_counts else None
    collision_occurred = "Yes" if collision_count > 0 or opponent_collision_count > 0 else "No"

    # Creating a dictionary with processed statistics
    processed_data = {
        'avg_reward': f"{avg_reward:.2f}",
        'avg_reward_10s': f"{avg_reward_10s:.2f}",
        'avg_progress': f"{avg_progress:.2f}",
        'avg_progress_10s': f"{avg_progress_10s:.2f}",
        'wrong_way_count': wrong_way_count,
        'wrong_way_percentage': f"{wrong_way_percentage:.2f}",
        'right_way_percentage': f"{right_way_percentage:.2f}",
        'collision_occurred': collision_occurred,
        'collision_count': collision_count
    }

    if multi:
        processed_data['opponent_collision_count'] = opponent_collision_count
        processed_data['most_frequent_rank'] = most_frequent_rank

    return processed_data
