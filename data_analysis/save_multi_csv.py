import csv
import os


def save_episode_to_csv(file_name, aggregated_data):
    """
    Save processed episode data (statistics) to a CSV file.
    :param file_name: Name of the CSV file.
    :param aggregated_data: List of dictionaries containing processed episode stats for each agent.
    """

    folder_name = os.path.join('..', 'csv_files')
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    true_file_name = os.path.join(folder_name, file_name)

    try:
        with open(true_file_name, mode='a', newline='') as file:
            if aggregated_data:
                keys = ['episode', 'agent'] + list(aggregated_data[0]['processed_data'].keys())
                writer = csv.DictWriter(file, fieldnames=keys)

                if file.tell() == 0:
                    writer.writeheader()

                for data in aggregated_data:
                    row = data['processed_data']
                    row['episode'] = data['episode']
                    row['agent'] = data['agent']
                    writer.writerow(row)

    except Exception as e:
        print(f"Error writing to CSV: {e}")
