import os
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping
from keras.metrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.models.main_model import ImageClassificationModel


class BasicCNN(ImageClassificationModel):
    def __init__(self, image_height, image_width, n_classes, model_results_path):
        super().__init__(image_height, image_width, n_classes, model_results_path, "basic_cnn")
        self.image_height = image_height
        self.image_width = image_height
        self.n_classes = n_classes

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.image_height, self.image_height, 3)))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.n_classes, activation='softmax'))  # Output layer with number of classes