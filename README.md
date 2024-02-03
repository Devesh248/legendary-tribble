# Image Classifier

This project contains an image classifier that classifies images in either of the classes, (Human, No Human, Uber Eats).

---------------------------------------------------------------------------------------------------------------------------
## Codebase
The Codebase is divided into following categories,

### Experiment Notebooks: 
This section contains notebooks with different model architecture like Basic CCN and Transfer Learning using VGG16. These notebooks also contain model architecture, different hyperparameters, accuracy and loss 
related graphs and confusion matrix related reports.

### Src: 
It contains all the functions and classes related to data processing, model training and predictions.

### App: 
This is a WebApp created by using the Streamlit python framework.

### Test images: 
Images to test the app and classification.


---------------------------------------------------------------------------------------------------------------------------
## Classes description:

There are 3 classes taken in consideration for training, Human, No Human, Uber Eats.
Number of samples of classes are,

Human: 733

No Human: 471

Uber Eats: 612 (After saving the augmented images)

---------------------------------------------------------------------------------------------------------------------------
## Cloud (AWS, Sagemaker)

### S3 Bucket:
Data has been stored into S3 buckets and fetched during model training, and once model is training done then the final model for each experiment is saved into S3 bucket.


### EC2:
A t2.medium instance has been used to deploy the app, which runs the Streamlit WebApp and performs image classification using the locally saved model.

### Sagmaker:
A model endpoint has been created for the inference using AWS sagemaker (Not used in running code, because of its cost). Current inference is happening by the locally saved model.

----------------------------------------------------------------------------------------------------------------------------
## Limitations:

### Webcam image capture:
It has the code and a scheduler to capture an image every 10 seconds, but it is able to capture only on the host system (if it has a webcam), but if the user is using Webapp on some other system (other than host), it won’t be able to capture it. To solve this issue we need to create a Webapp client using some frontend libraries, like javascript etc.

### Speed and Latency: 
The application will take some time to load up the screen, and also for the first classification request it will be a bit slow (because of model loading), but for subsequent requests it will be fine.

### Security:
Current Webapp is running on HTTP, so if you are not able to open it, try using HTTP only not HTTPS. 

