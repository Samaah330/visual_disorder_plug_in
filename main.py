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

def categorize_ratings(rating):
    '''
        replace ratings into catergories

        :param rating:
        :return: new rating
    '''
    if (rating < 2.69):
        rating = 0
    elif (rating >= 2.69 and rating < 4.06):
        rating = 1
    elif (rating >= 4.06 and rating < 5.36):
        rating = 2
    elif (rating >= 5.36):
        rating = 3

    return rating

def get_dataframe():
    df = pd.read_excel(
        r'C:\Users\SamaahMachine\Documents\Argonne\Images with Ratings\normalized_training_data.xlsx')

    plugin.publish('Data:', df)

    return df

def get_accuracy(accuracy):
    return accuracy


def process_frame(frame):

    df = get_dataframe()

    # categorize ratings
    df['Order'] = df['Order'].apply(categorize_ratings)

    df = shuffle(df)

    # assign 80 percent of the data as training data,
    # assign 20 percent of the data for testing later
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.80

    # creating dataframes with test rows and training rows
    train = df[df['is_train'] == True]
    test = df[df['is_train'] == False]

    features = df.columns[2:11]
    print(features)

    features_train = train[features]
    labels_train = train['Order']
    features_test = test[features]
    labels_test = test['Order']

    # Creating a random forest classifier
    model = RandomForestClassifier(n_estimators=2000, max_features="log2", criterion="entropy", min_samples_split=8)

    # Training the classifier
    model.fit(features_train, labels_train)

    accuracy = model.score(features_test, labels_test)

    plugin.publish('Accuracy:', accuracy)

    return model.predict(frame)

def main():
    plugin.init()

    cam = Camera(format = BGR)

    # for each image it captures in camera_stream
    for sample in cam.stream():
        results = process_frame(sample.data)
        logging.info("results %s", results)
        time.sleep(60)

if __name__ == "__main__":
    main()