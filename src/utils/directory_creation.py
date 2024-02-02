from src.utils.constants import *


def create_directory():
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    if not os.path.exists(webcam_data_path):
        os.makedirs(webcam_data_path)

    if not os.path.exists(original_data_path):
        os.makedirs(original_data_path)

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path)

