import csv
import os


def save_episode_to_csv(file_name, processed_data):
    """
    Save processed episode data (statistics) to a CSV file.
    :param file_name: Name of the CSV file.
    :param processed_data: Dictionary containing processed episode stats.
    """

    folder_name = os.path.join('..', 'csv_files')
    os.makedirs(folder_name, exist_ok=True)  # Creating the folder if it doesn't exist
    true_file_name = os.path.join(folder_name, file_name)

    try:
        with open(true_file_name, mode='a', newline='') as file:
            keys = processed_data.keys()
            writer = csv.DictWriter(file, fieldnames=keys)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(processed_data)

    except Exception as e:
        print(f"Error writing to CSV: {e}")
