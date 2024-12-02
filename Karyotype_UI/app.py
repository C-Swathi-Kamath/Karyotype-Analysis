from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import pickle
import cv2
import joblib
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def loadPicklemodel(modelPath):
    loaded_model = joblib.load(modelPath)
    return loaded_model

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

xgBoostPath = 'C:\\Users\\ashwi\\Desktop\\New folder\\Karyotype\\xgboost1.pkl'
rfPath = 'RF1.pkl'
NBPath = 'NB1.pkl'
svmPath = 'SVM1.pkl'
incPath = 'inception62.h5'
# Load your trained model
model = load_model(incPath)
#model.make_predict_function()          

with open(xgBoostPath, 'rb') as f:
    xgModel =  pickle.load(f)
rfModel = joblib.load(rfPath)
NBModel = joblib.load(NBPath)
svmModel = joblib.load(svmPath)

# Class labels
class_labels = [ 'Down Syndrome','Normal Female', 'Normal Male','CML']


def predictPickleModel(image_path, model):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    img = np.repeat(img, 3, axis=2)
    img = preprocess_input(img)
    img_features = np.reshape(img, (1, -1))
    predicted_class = model.predict(img_features)[0]
    return predicted_class

def model_predict(img_path, model):
    # Preprocessing the image
    img = cv2.resize(cv2.imread(img_path),(224,224))
    img_normalized = img/255
    preds = np.argmax(model.predict(np.array([img_normalized])))
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print('File path:', file_path)

        result = model_predict(file_path, model)
        print("inceptionV3 results = ", result)
        img = Image.open(file_path).convert('L')  # Convert image to grayscale
        img = img.resize((299, 299))
        img = np.array(img)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
        img = preprocess_input(img)
        img_features = np.reshape(img, (1, -1))

        rfPred = rfModel.predict(img_features)[0]
        rfPred = int(rfPred)
        svmPred = svmModel.predict(img_features)[0]
        svmPred = int(svmPred)
        xgPred = xgModel.predict(img_features)[0]
        xgPred = int(xgPred)
        NBPred = NBModel.predict(img_features)[0]
        NBPred = int(NBPred)
        # Process your result for class
        xgResults = class_labels[xgPred]
        rfResults = class_labels[rfPred]
        nBResults = class_labels[NBPred]
        svmResults = class_labels[svmPred]
        incResults = class_labels[result]
        return jsonify({'result': str(incResults),
                'xgResults': str(xgResults),
                'rFResults': str(rfResults),
                'nBResults': str(nBResults),
                'svmResults': str(svmResults)})


    return None

if __name__ == '__main__':
    app.run(debug=True)
