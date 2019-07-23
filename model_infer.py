from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix
from pathlib import Path

import tensorflow as tf
import numpy as np
import os
import cv2

if __name__ == '__main__':

    # ignore annoying tf warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # define model variables
    class_dict = {0:'closed', 1:'open'}
    img_width, img_height = 256, 192
    img_shape = (img_width, img_height, 3)

    test_dir = os.path.realpath('data/test/')
    weights_file = 'boot_model_weights.h5'

    # reconstruct model and load saved weights
    print('Reconstructing model')

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

    model.load_weights(weights_file)

    # load test images and manipulate np arrays...
    print('Running test image set')
    test_imgs, y_true, y_predict, predicted_classes = [], [], [], {}

    # grab files
    all_test_files = []
    for subdir, dirs, files in os.walk(test_dir):
        for file in files:
            all_test_files.append(os.path.join(subdir, file))

    # loop through files and run through model
    for i in range(len(all_test_files)):
        # load img and scale for model
        y_true.append(os.path.basename(Path(all_test_files[i]).parent))

        img = cv2.imread(all_test_files[i])
        test_imgs.append(img)
        img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) / 255
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)

        # run single prediction
        predict = model.predict(np.expand_dims(img, axis=0))
        predict = np.round(predict, 0)[0, 0]
        y_predict.append(class_dict.get(predict))
        predicted_classes[os.path.basename(all_test_files[i])] = class_dict.get(predict)

        print(str(i+1) + '/' + str(len(all_test_files)) + ': ' + os.path.basename(all_test_files[i])
              + ' ' + predicted_classes[os.path.basename(all_test_files[i])])

    # prediction results
    cm = confusion_matrix(y_true, y_predict)
    print(cm)

    # keep track of wrong prediction indexes
    wrong_predictions = []
    for i in range(len(y_predict)):
        if y_predict[i] != y_true[i]:
            wrong_predictions.append(i)
            break

    # display each wrong image
    for i in range(len(wrong_predictions)):
        img = test_imgs[wrong_predictions[i]]
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        cv2.putText(img, y_predict[wrong_predictions[i]], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(str(i+1) + '/' + str(len(wrong_predictions)), img)

    cv2.waitKey(0)

"""
    # display 5 random images with their predicted class
    rand_files = random.sample(all_test_files, 5)
    for file in rand_files:
        img = cv2.imread(file)
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        cv2.putText(img, predicted_classes[os.path.basename(file)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(file, img)
        
    cv2.waitKey(0)
"""