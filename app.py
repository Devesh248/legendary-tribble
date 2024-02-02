from PIL import Image
import streamlit as st
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from src.utils.constants import image_height, image_width
from src.utils.directory_creation import create_directory
from src.main import ImageClassification
from src.utils.webcam_capture import capture_image

supported_image_format = ['png', 'jpeg', 'jpg']


def main(classifier_obj):
    st.title("Image Classifier")
    uploaded_file = st.file_uploader("Choose an image...", type=supported_image_format)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if uploaded_file.name.split('.')[-1].lower() not in supported_image_format:
            st.image(image)
            st.text("Invalid image format ..")

        image = image.resize((image_height, image_width))
        response = classifier_obj.classify_image(np.array(image))
        st.image(image)
        if response['human_detected'] == 'Human Detected':
            st.text(response['human_detected'])
            st.text(response['delivery'])
        else:
            st.text(response['human_detected'])


if __name__ == "__main__":
    create_directory()
    classifier_obj = ImageClassification()

    scheduler = BackgroundScheduler()
    scheduler.add_job(capture_image, 'interval', seconds=10, args=(classifier_obj, ))
    scheduler.start()

    main(classifier_obj)