import cv2
from src.utils.constants import *
from src.models.basic_cnn import BasicCNN
from src.models.transfer_learning_vgg16 import TransferLearningVGG16
from src.data_processing.data_preparation import DataPreparation

model = None


class ImageClassification:
    def __init__(self):
        self.data_handler = DataPreparation()

        if model_type == "basic_cnn":
            self.classification_model = BasicCNN(image_height, image_width, n_classes, model_results_path)
        elif model_type == "transfer_learning_vgg16":
            self.classification_model = TransferLearningVGG16(image_height, image_width, n_classes, model_results_path)
        else:
            raise Exception("Not a valid model type")

    def prepare_data(self):
        train_generator, test_generator, val_generator = self.data_handler.prepare_data()
        return train_generator, test_generator, val_generator

    def train_model(self, train_generator, val_generator):
        classifier_model = self.classification_model.get_model()
        n_steps = train_generator.samples // batch_size
        n_val_steps = val_generator.samples // batch_size

        self.classification_model.train_model(n_steps, n_val_steps, batch_size, n_epochs, train_generator, val_generator, classifier_model)

    def classify_image(self, image):
        image = cv2.resize(image, (image_width, image_height))
        predicted_label, prediction_prob = self.classification_model.predict(class_map, image)
        response = dict()
        if predicted_label == "human":
            response['human_detected'] = "Human Detected"
            response['delivery'] = "others"
        elif predicted_label == "no_human":
            response['human_detected'] = "No Human Detected"
            response['delivery'] = None
        elif predicted_label == "uber_eats":
            response['human_detected'] = "Human Detected"
            response['delivery'] = "uber_eats"

        return response







