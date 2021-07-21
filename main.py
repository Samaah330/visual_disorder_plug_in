import argparse
import logging
import time
import numpy as np
from waggle import plugin
from waggle.data.vision import Camera
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from waggle.data.vision import Camera, ImageFolder, RGB, BGR




def categorize_ratings(rating):
    '''
        replace ratings into catergories ( can add more categories once this works)
        [1, 3)   --> 0    (most disorderly)
        [3, 5) --> 1
        [5, 7) --> 2      (most orderly)

        :param rating:
        :return: new rating
    '''

    if (rating >= 1 and rating < 4):
        rating = 0
    elif (rating >= 4 and rating < 6):
        rating = 1
    elif (rating >= 6 and rating < 7):
        rating = 2

    return rating

def process_frame(frame):
    df = pd.read_excel(
        r'C:\Users\SamaahMachine\Documents\Argonne\Images with Ratings\training_data.xlsx')

    # categorize ratings
    df['Order'] = df['Order'].apply(categorize_ratings)

    # normalize all values
    column_names = ['SED', 'Entropy', 'sdValue', 'sdSat', 'sdHue', 'Mean Value', 'Mean Hue', 'Mean Sat', 'ED']

    for i in range(len(column_names)):
        df[column_names[i]] = (df[column_names[i]] - df[column_names[i]].min()) / (df[column_names[i]].max() - df[column_names[i]].min())

    # shuffle data set so that the training and test data can have a mix of all labels
    df = shuffle(df)

    # assign 80 percent of the data as training data,
    # assign 20 percent of the data for testing later
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.80

    # creating dataframes with test rows and training rows
    train = df[df['is_train'] == True]
    test = df[df['is_train'] == False]


    del df['ED']
    features = df.columns[2:10]
    print(features)

    features_train = train[features]
    labels_train = train['Order']

    model = RandomForestClassifier(n_estimators = 1000, max_features = "log2", criterion = "entropy", min_samples_split = 6)

    # Training the classifier
    model.fit(features_train, labels_train)

    return model.predict(frame)

def main():
    plugin.init()

    cam = Camera(format = BGR)

    for sample in cam.stream():
        results = process_frame(sample.data)
        logging.info("results %s", results)
        time.sleep(60)


if __name__ == "__main__":
    main()