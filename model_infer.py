from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from sklearn.metrics import confusion_matrix
from pathlib import Path

import tensorflow as tf
import numpy as np
import cv2 as cv
import os, random

if __name__ == '__main__':

    RUN_MODEL_ON_TEST_DATA = True
    TEST_DATA_SORTED_BY_CLASS = False
    CV_DISPLAY_WRONG_PREDICTIONS = True # works if TEST_DATA_SORTED_BY_CLASS = TRUE
    CV_DISPLAY_SOME_RAND_PREDICTIONS = True

    test_dir = os.path.realpath('test_data/')
    weights_filename = 'boot_model_weights.h5'

    # ignore annoying tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # match model predictions to corresponding class
    class_dict = {0:'closed', 1:'open'}

    # define model input parameters
    img_width, img_height = 256, 192
    img_shape = (img_width, img_height, 3)

    if RUN_MODEL_ON_TEST_DATA:
        # reconstruct model and load saved weights
        print('RUN_MODEL_ON_TEST_DATA = TRUE, Reconstructing model and predicting on test set')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=img_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.load_weights(weights_filename)

        # indexes for each list will match
        true_classes, file_paths, predicted_classes = [], [], []

        # grab file paths and store in list
        for subdir, dirs, files in os.walk(test_dir):
            for file in files:
                file_paths.append(os.path.join(subdir, file))

        # open file to save model predictions to txt
        output_txt = open("output.txt", "w+")

        # loop through files and run them through model
        for i in range(len(file_paths)):
            # store true class of image by getting its parent directory
            true_classes.append(os.path.basename(Path(file_paths[i]).parent))

            # load img and scale for model
            img = cv.imread(file_paths[i])
            img = np.float32(cv.cvtColor(img, cv.COLOR_BGR2RGB)) / 255
            img = cv.resize(img, (img_height, img_width), interpolation=cv.INTER_AREA)

            # run single prediction
            model_prediction = model.predict(np.expand_dims(img, axis=0))

            # round to nearest 0 or 1 and retrieve value from np array
            model_prediction = np.round(model_prediction, 0)[0, 0]

            # store class prediction by referring back to class definition dictionary
            predicted_classes.append(class_dict.get(model_prediction))

            newline = str(i+1) + '/' + str(len(file_paths)) + ': ' + file_paths[i] \
                      + ' ' + predicted_classes[i]
            print(newline)
            output_txt.write(newline+"\n")

        output_txt.close()

    # if we didn't just run through test set, we need to read predictions from previous output file
    else:
        print('RUN_MODEL_ON_TEST_DATA = FALSE, getting model predictions from previous output.txt\n')

        output_text = open('output.txt', 'r')
        num_lines = int(len(output_text.readlines()))
        output_text.close()
        output_text = open('output.txt', 'r')

        true_classes, file_paths, predicted_classes = [], [], []
        for i in range(num_lines):
            line = output_text.readline().replace('\n', '')
            print(line)

            # each line is organized by [index/total_imgs, file_path, class_prediction]
            split = line.split(' ')

            # store file path and class, indexes match in separate lists
            file_paths.append(split[1])
            predicted_classes.append(split[2])
            true_classes.append(os.path.basename(Path(file_paths[i]).parent))

    if TEST_DATA_SORTED_BY_CLASS:
        # print results of model predictions (confusion matrix)
        cm = confusion_matrix(true_classes, predicted_classes)
        print()
        print(cm)
        print()

        # store index of each wrong prediction
        wrong_prediction_indexes = []
        for i in range(len(predicted_classes)):
            if predicted_classes[i] != true_classes[i]:
                wrong_prediction_indexes.append(i)

        if CV_DISPLAY_WRONG_PREDICTIONS:
            # display each wrongly predicted image
            print('CV_DISPLAY_WRONG_PREDICTIONS = TRUE, Displaying true negatives and false positives...')
            for i in wrong_prediction_indexes:
                img = cv.imread(file_paths[i])
                img = np.float32(cv.cvtColor(img, cv.COLOR_BGR2RGB)) / 255
                img = cv.resize(img, (img_width, img_height), interpolation=cv.INTER_AREA)
                cv.putText(img, predicted_classes[i], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow(str(i+1) + '/' + str(len(file_paths)), img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    if CV_DISPLAY_SOME_RAND_PREDICTIONS:
        # display 5 random images with their predicted class
        print('CV_DISPLAY_SOME_RAND_PREDICTIONS = TRUE, displaying 5 random images with predicted class result...')
        for i in range(5):
            rand_index = random.randint(0, len(file_paths))
            img = cv.imread(file_paths[rand_index])
            img = cv.resize(img, (img_width, img_height), interpolation=cv.INTER_AREA)
            cv.putText(img, predicted_classes[rand_index], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.imshow(file_paths[rand_index], img)
        cv.waitKey(0)
