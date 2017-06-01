from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.utils import shuffle
import numpy as np
import sklearn
import csv
import cv2

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# Correction for left & right camera of the vehicle (see in generator())
correction = 0.1


def generator(samples, batch_size=32):
    """
    Using Keras generator for optimization
    :param samples: sample images
    :param batch_size: batch size
    :return: yeild next portion of images
    """
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):  # Using all 3 cameras
                    name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    if i == 1:  # left camera correction
                        center_angle += correction
                    elif i == 2:  # right camera correction
                        center_angle -= correction

                    images.append(center_image)
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle)
                    angles.append(center_angle * -1.0)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Dimensions of the images
ch, row, col = 3, 160, 320

# Using NVIDIA CNN architecture (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((60, 25), (0, 0))))  # Cropping the non-road pixels
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Using mse because it gives the best results
# Found out that more that 2 epochs is overfitting
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 6,  # because of data augmentation
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=2)

model.save('model.h5')
