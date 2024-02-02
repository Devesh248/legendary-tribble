import os
import splitfolders
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from src.utils.constants import *
from src.utils.directory_creation import create_directory
from src.data_processing.data_processing import DataProcessing


class DataPreparation:
    def __init__(self):
        create_directory()
        self.data_processor = DataProcessing()

    def preprocess_data(self, input_data_path, output_data_path, num_augmentations):
        self.data_processor.augment_and_save(input_data_path, output_data_path, num_augmentations)

    @staticmethod
    def data_split(input_data_path, output_data_path, train_size, test_size, validation_size):
        splitfolders.ratio(input_data_path, output_data_path, ratio=(train_size, validation_size, test_size))

    def create_data_generators(self):
        datagen = ImageDataGenerator(
            rescale=1. / 255,  # Rescale pixel values to 0-1
            rotation_range=40,  # Rotate images randomly by up to 40 degrees
            width_shift_range=0.2,  # Shift images horizontally by up to 20%
            height_shift_range=0.2,  # Shift images vertically by up to 20%
            shear_range=0.2,  # Shear images randomly
            zoom_range=0.2,  # Zoom images randomly
            horizontal_flip=True,  # Flip images horizontally randomly
            fill_mode='nearest')  # Fill empty pixels with nearest neighbor interpolation

        train_generator = datagen.flow_from_directory(
            os.path.join(split_data_path, "train"),
            target_size=(image_height, image_width),  # Resize images to a consistent size
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')

        test_generator = datagen.flow_from_directory(
            os.path.join(split_data_path, "test"),
            target_size=(image_height, image_width),  # Resize images to a consistent size
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')

        val_generator = datagen.flow_from_directory(
            os.path.join(split_data_path, "val"),
            target_size=(image_height, image_width),  # Resize images to a consistent size
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')

        return train_generator, test_generator, val_generator

    def prepare_data(self, num_augmentations=3):
        for cls in os.listdir(original_data_path):
            input_data_path = os.path.join(original_data_path, cls)
            output_data_path = os.path.join(processed_data_path, cls)
            self.preprocess_data(input_data_path, output_data_path, num_augmentations)

        self.data_split(processed_data_path, split_data_path, 0.8, 0.1, 0.1)

        train_generator, test_generator, val_generator = self.create_data_generators()

        return train_generator, test_generator, val_generator


