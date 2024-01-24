import os
import pickle
import json
import numpy as np
import argparse
import warnings
from keras.models import load_model

warnings.filterwarnings('ignore') 

from mputils import FaceDetector

face_detector = FaceDetector('models/face_landmarker.task')

# load logreg model
with open("models/logreg.sav", "rb") as model_file:
    logreg_model = pickle.load(model_file)

# load nn model
nn_model = load_model("models/nn_model_25columns.h5")

# Convert image to feature vectors
def extract_features_from_image(rgb_image):
    _, _, areas_histogram = face_detector.image_pipeline(rgb_image)
    feature_vector = np.array(list(areas_histogram.values())).flatten()
    feature_vector = feature_vector.reshape(1, -1)
    return feature_vector

# Make a prediction on the feature vector
def make_prediction(feature_vector,
                    model_type = "logreg",
                    return_type="proba"):
    
    if model_type == "logreg":
        if return_type == "class":
            return logreg_model.predict(feature_vector)[0]
        elif return_type == "proba":
            return round(logreg_model.predict_proba(feature_vector)[0][1], 3)
    elif model_type == "nn":
        prediction = nn_model.predict(feature_vector)
        if return_type == "class":
            return np.argmax(prediction, axis=1)[0]
        elif return_type == "proba":
            return round(float(prediction[0][1]), 3)


# Images from folder to list of arrays 
def read_folder(dir_path):
    image_names = [name for name in os.listdir(dir_path) if not name.startswith(".")]
    images = []
    for name in image_names:
        image_path = os.path.join(dir_path, name)
        rgb_image = face_detector.read_image(image_path)
        images.append(rgb_image)
    return images, image_names

# Batch predict (works with list of np.arrays)
def bath_predict(images, model_type):
    predictions = []
    for rgb_image in images:
        feature_vector = extract_features_from_image(rgb_image)
        prediction = make_prediction(feature_vector, model_type)
        predictions.append(prediction)
    return predictions


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Classification Script")
    parser.add_argument("dir_path", type=str, help="Directory path containing images to classify")
    parser.add_argument("model_type", type=str, help="logreg or nn")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    images, image_names = read_folder(args.dir_path)
    predictions = bath_predict(images, args.model_type)
    with open("prediction.json", 'w') as f:
        pairs = dict(zip(image_names, predictions))
        print(pairs)
        json.dump(pairs, f)
    print("Prediction saved in prediction.json file")