import argparse
import logging
import time
import numpy as np
from waggle import plugin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from waggle.data.vision import Camera, ImageFolder, RGB, BGR
import pickle
from random_forest_model import create_random_forest

def publish_dataframe():
    df = pd.read_excel(
        r'C:\Users\SamaahMachine\Documents\Argonne\Images with Ratings\normalized_training_data.xlsx')

    plugin.publish('Data:', df)

def publish_accuracy(model):
    test_data = pd.read_excel(
        r'C:\Users\SamaahMachine\Documents\Argonne\Images with Ratings\test_data.xlsx')

    features = test_data.columns[2:11]

    features_test = test_data[features]
    labels_test = test_data['Order']

    accuracy = model.fit(features_test, labels_test)

    plugin.publish("Accuracy: ", accuracy)

def process_frame(model, frame):

    return model.predict(frame)

def main():
    plugin.init()

    # uncomment to load in dataframe this way instead
    # with open(r'C:\Users\SamaahMachine\Documents\Argonne\Images with Ratings\random_forest_model.pkl', 'rb') as f
    #     model = pickle.load(f)

    model = create_random_forest()

    publish_dataframe()

    publish_accuracy(model)

    cam = Camera(format = BGR)

    # for each image it captures in camera_stream
    for sample in cam.stream():
        results = process_frame(model, sample.data)
        logging.info("results %s", results)
        time.sleep(60)

    # end goal - create a map of disorder
    # should I be publishing every image that I classified?

if __name__ == "__main__":
    main()