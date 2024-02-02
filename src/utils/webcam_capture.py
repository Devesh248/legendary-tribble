import os
import cv2
import time
from src.utils.logger import logger
from src.utils.constants import webcam_data_path


def capture_image(classifier_obj):
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        current_timestamp = int(time.time())
        image_name = f"captured_image_{current_timestamp}.png"
        image_path = os.path.join(webcam_data_path, image_name)
        response = classifier_obj.classify_image(frame)
        if response['human_detected'] == 'Human Detected':
            if response['delivery'] == 'uber_eats':
                logger.info(f"Image: {image_name}, Label: Human Detected, Uber Eats")
            else:
                logger.info(f"Image: {image_name}, Label: Human Detected, Other")
        else:
            logger.info(f"Image: {image_name}, Label: No Human Detected")

        cv2.imwrite(image_path, frame)

    except:
        pass
