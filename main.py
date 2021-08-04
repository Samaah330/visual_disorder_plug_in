import argparse
import logging
import time
import numpy as np
from waggle import plugin
import pandas as pd
from waggle.data.vision import Camera, ImageFolder, RGB, BGR
import pickle
from random_forest_model import create_random_forest
import cv2
from detect_features import find_mean_hsv
from detect_features import find_standard_deviation_hsv
from detect_features import find_edge_density
from detect_features import find_straight_edge_density
from detect_features import find_entropy

def publish_dataframe():
    '''
    publishes dataframe
    :return: void
    '''
    df = pd.read_excel(r'normalized_data.xlsx')

    plugin.publish('Data:', df)

def publish_accuracy(model):
    '''
    Computes accuracy based on test data and publishes it
    :param model: random forest model
    :return: void
    '''
    test_data = pd.read_excel(r'test_data.xlsx')

    features = test_data.columns[2:11]

    features_test = test_data[features]
    labels_test = test_data['Order']

    accuracy = model.fit(features_test, labels_test)

    plugin.publish("Accuracy: ", accuracy)

def process_frame(model, frame):
    '''
    Given an image, predicts order rating based on computed features
    :param model: random forest model used to predict order rating
    :param frame: image retrieved from Sage nodes
    :return: void
    '''
    predictions_df = pd.read_excel(r'base.xlsx')

    image = frame
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mean_hsv = find_mean_hsv(hsv_image)
    standard_dev_hsv = find_standard_deviation_hsv(hsv_image)
    edge_density = find_edge_density(image)
    straight_edge_density = find_straight_edge_density(image)
    image = frame
    entropy = find_entropy(image)

    # replace values with the features that computed
    predictions_df.loc[0, 'Mean Hue'] = mean_hsv[0]
    predictions_df.loc[0, 'Mean Sat'] = mean_hsv[1]
    predictions_df.loc[0, 'Mean Value'] = mean_hsv[2]
    predictions_df.loc[0, 'sdHue'] = standard_dev_hsv[0]
    predictions_df.loc[0, 'sdSat'] = standard_dev_hsv[1]
    predictions_df.loc[0, 'sdValue'] = standard_dev_hsv[2]
    predictions_df.loc[0, 'ED'] = edge_density
    predictions_df.loc[0, 'SED'] = straight_edge_density
    predictions_df.loc[0, 'Entropy'] = entropy

    prediction = model.predict(predictions_df)

    print(prediction[0])

    plugin.publish('Prediction:', prediction[0])

def main():
    '''
    publishes dataframe, accuracy, and prediction for each image
    retrieved from Sage nodes
    :return: void
    '''
    plugin.init()

    # uncomment to load in dataframe this way instead
    # with open(r'random_forest_model.pkl', 'rb') as f
    #     model = pickle.load(f)

    model = create_random_forest()

    publish_dataframe()

    publish_accuracy(model)

    cam = Camera(format = BGR)

    for sample in cam.stream():
        results = process_frame(model, sample.data)
        logging.info("results %s", results)
        time.sleep(60)

if __name__ == "__main__":
    main()