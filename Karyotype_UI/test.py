import joblib
import os
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# Load the model from the file
loaded_model = joblib.load('XG2.pkl')
def predict_single_image(image_path, model):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((299, 299))  # Resize the image to (299, 299)
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    img = np.repeat(img, 3, axis=2)
    img = preprocess_input(img)
    img_features = np.reshape(img, (1, -1))  # Reshape the image to match the feature matrix shape

    # Predict the class label
    predicted_class = model.predict(img_features)[0]

    return predicted_class

c ="NM"
class_labels = [ 'Down Syndrome','Normal Female', 'Normal Male','CML']
folder_path = f'dataset/{c}'

# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    # Get the full path of the image file
    image_path = os.path.join(folder_path, filename)

    # Predict the class label for the image
    predicted_class = predict_single_image(image_path, loaded_model)

    # Print the image name and predicted class
    print(f"Image: {filename}, Predicted Class: {class_labels[predicted_class]}")