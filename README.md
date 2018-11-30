# American Sign Language Interpreter using Deep Learning and Python
American Sign Language is a way of expressing characters through hand gestures
(more info : https://en.wikipedia.org/wiki/American_Sign_Language).

The project aims to create a model that can take the camera feed and predict the sign displayed by the person. I have used Keras to create a conv net followed by dense laters to create the model. The model is trained on ASL dataset available on kaggle.(dataset:https://www.kaggle.com/grassknoted/asl-alphabet )

The trained model has a test accuracy of ~95% and is saved as 'asl_language.d5'.

sign_classifier.ipynb contains the model code. ASL_detector uses OpenCV and trained model to make live predictions.
