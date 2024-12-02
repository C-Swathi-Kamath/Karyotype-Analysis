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

dataset_dir = 'dataset/'
class_names = list(sorted(os.listdir(dataset_dir)))

# Define the numeric labels based on the order of class names
label_map = {class_name: label for label, class_name in enumerate(class_names)}

image_paths = []
labels = []
# Iterate over the class folders
for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        image_files = os.listdir(class_dir)
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            image_paths.append(image_path)
            labels.append(label_map[class_name])

X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Convert images to grayscale and preprocess them
train_features = []
for img_path in X_train:
    img = Image.open(img_path).convert('L')  # Convert image to grayscale
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    img = np.repeat(img, 3, axis=2)
    img = preprocess_input(img)
    #img = model.predict(np.array([img]))  # Pass image as a batch of size 1
    train_features.append(img)

test_features = []
for img_path in X_test:
    img = Image.open(img_path).convert('L')  # Convert image to grayscale
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    img = np.repeat(img, 3, axis=2)
    img = preprocess_input(img)
   # img = model.predict(np.array([img]))  # Pass image as a batch of size 1
    test_features.append(img)

# Reshape the feature matrices
X_train_features = np.reshape(train_features, (len(train_features), -1))
X_test_features = np.reshape(test_features, (len(test_features), -1))

# Train XGBoost classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_features, y_train)


import joblib
joblib.dump(xgb_classifier, 'XG1.pkl')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report

y_pred = xgb_classifier.predict(X_test_features)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names, rotation=0)
plt.show()

report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        image_files = os.listdir(class_dir)
        num_images = len(image_files)
        print(f"Class '{class_name}': {num_images} images")

