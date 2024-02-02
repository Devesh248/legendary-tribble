import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback
from src.utils.logger import logger

classification_predictor = None


class ImageClassificationModel:
    def __init__(self, image_height, image_width, n_classes, model_result_path, model_type):
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes
        self.model_type = model_type
        self.model_result_path = model_result_path

    def train_model(self, n_steps, n_val_steps, batch_size, n_epochs, train_generator, val_generator, model):
        plot_loss_1 = PlotLossesCallback()

        # ModelCheckpoint callback - save best weights
        tl_checkpoint_1 = ModelCheckpoint(filepath=os.path.join(self.model_result_path, 'tl_model_v1.weights.best.hdf5'),
                                          save_best_only=True,
                                          verbose=1)

        # EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   mode='min')

        history = model.fit(train_generator,
                            batch_size=batch_size,
                            epochs=n_epochs,
                            validation_data=val_generator,
                            steps_per_epoch=n_steps,
                            validation_steps=n_val_steps,
                            callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                            verbose=1)

    def predict(self, class_map, image):
        global classification_predictor
        if classification_predictor is None:
            logger.info("Loading new model ..")
            classification_predictor = load_model(os.path.join(self.model_result_path, 'tl_model_v1.weights.best.hdf5'))

        # Load the image
        # img = load_img(image_path, target_size=(self.image_height, self.image_width))
        # img_array = image

        # Preprocess the image
        img_batch = np.expand_dims(image, axis=0)
        img_batch = img_batch / 255.0  # Normalize to [0, 1] range

        # Make predictions
        predictions = classification_predictor.predict(img_batch, verbose=0)[0]

        pred_class = predictions.argmax()
        pred_prob = predictions[pred_class]

        pred_label = class_map[pred_class]

        return pred_label, pred_prob

