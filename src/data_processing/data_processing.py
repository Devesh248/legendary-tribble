import os
import cv2
from imgaug import augmenters as iaa


class DataProcessing:
    def __init__(self):
        pass

    @staticmethod
    def augment_and_save(input_data_path, output_data_path, num_augmentations=1):
        """Augments an image and saves the augmented versions to files.

        Args:
            input_data_path (str): input path from where images need to be processed.
            output_data_path (str): output path where processed images need to be stored.
            num_augmentations (int, optional): Number of augmented images to generate. Defaults to 1.

        Raises:
            ValueError: If the output directory does not exist or cannot be created.
        """
        for image_name in os.listdir(input_data_path):
            image_path = os.path.join(input_data_path, image_name)
            try:
                # Load the image
                image = cv2.imread(image_path)
                cv2.imwrite(image_path, image)
                # Check if the output directory exists, create it if necessary
                if not os.path.exists(output_data_path):
                    os.makedirs(output_data_path)

                # Define augmentation sequence using imgaug
                augmentations = iaa.Sequential([
                    iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
                    iaa.Flipud(0.5),  # Vertical flip with 50% probability
                    iaa.Crop(percent=(0.05, 0.2)),  # Random cropping within 5-20% of the image area
                    iaa.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1)),  # Affine transformations
                    iaa.Add((-10, 10)),  # Pixel value adjustments
                    iaa.ElasticTransformation(alpha=0.5, sigma=0.25),  # Elastic distortions
                ])

                # Generate augmented images
                augmented_images = augmentations.augment_images([image] * num_augmentations)

                # Save the augmented images
                for i, aug_image in enumerate(augmented_images):
                    output_path = os.path.join(output_data_path, f"{image_name.split('.')[0]}_{i}.png")
                    cv2.imwrite(output_path, aug_image)
            except Exception as e:
                print(f"{image_name.split('.')[0]}: {e}")

