import os

home_path = os.path.abspath(os.curdir)

model_type = "transfer_learning_vgg16"
image_height = 224
image_width = 224
n_classes = 3

class_map = {0: 'human', 1: 'no_human', 2: 'uber_eats'}

log_file_path = os.path.join(home_path, "logs")

webcam_data_path = os.path.join(home_path, "data", "webcam")
original_data_path = os.path.join(home_path, "data", "original")
processed_data_path = os.path.join(home_path, "data", "processed")
split_data_path = os.path.join(home_path, "data", "split")

model_results_path = os.path.join(home_path, "results", "models")

batch_size = 32
n_epochs = 100

