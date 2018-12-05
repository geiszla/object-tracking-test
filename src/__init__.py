"""Main module

This module trains a machine learning model for tracking objects on video.

"""

from os import environ
from time import sleep

import matplotlib.pyplot as pyplot
from pandas import option_context
from skimage.io import imread
from skimage.measure import compare_ssim
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data import show_object_location, get_data_from_directory
from sliding_window import SlidingWindowDetector


if __name__ == '__main__':
    _init()


def _init():
    # Disable AVX2 warning
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Read in a set of image data
    # person_data = get_data_from_directory('Kwon_VTD\\soccer')
    # (person_data, data_directory) = get_data_from_directory('BoBot\\Vid_J_person_floor')

    # # Print all rows of data
    # with option_context('display.max_rows', 999):
    #     print(person_data)

    # _show_samples(person_data)

    # Create sliding-window detector instance
    positive_example = imread('blended.jpeg')
    negative_example = imread('trn_174-roi.jpeg')
    detector = SlidingWindowDetector(positive_example, negative_example)

    # Create model and use it to detect object location on image
    model = _create_mobilenet_model()
    print(model.summary())
    detector.detect_object(positive_example, model)

    # Print detection statistics
    (recall, f1_score) = detector.get_statistics()
    print("The recall of the model:", recall)
    print("The F1 score of the model:", f1_score)

    input('Press (almost) any key to exit.')

def _create_mobilenet_model():
    # Create pre-trained model from weights
    local_weights_file = 'model_data/ \
      mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    pre_trained_model = MobileNetV2(input_shape=(140, 140, 3), include_top=False, weights=None)
    pre_trained_model.load_weights(local_weights_file)

    # Don't train pre-trained model weights
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Get intermediate layer for input to our fully-connected layers
    last_layer = pre_trained_model.get_layer('block_13_expand_relu')
    print('last layer output shape:', last_layer.output_shape)

    # Append our fully-connected layers
    custom_layers = layers.Flatten()(last_layer.output)
    custom_layers = layers.Dense(1024, activation='relu')(custom_layers)
    custom_layers = layers.Dropout(0.2)(custom_layers)
    custom_layers = layers.Dense(1, activation='sigmoid')(custom_layers)

    # Configure and compile the model
    model = Model(pre_trained_model.input, custom_layers)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001),
        metrics=['acc'])

    return model

#     history = model.fit_generator(training_generator, steps_per_epoch=100, epochs=2,
#       validation_data=validation_generator, validation_steps=50, verbose=2)

def _show_samples(person_data):
    # Select a sample image and show it
    _, (axis1, axis2) = pyplot.subplots(1, 2, figsize=(20, 10))

    _, test_row = next(person_data.sample(1, random_state=2018).iterrows())
    original_image = imread(test_row['path'])
    axis1.imshow(original_image)

    processed_image = show_object_location(original_image, test_row)
    axis2.imshow(processed_image)

    # Show location of object on set of frames
    figure, axis_tuple = pyplot.subplots(5, 5, figsize=(30, 25))
    for current_axis, (_, test_row) in zip(axis_tuple.flatten(), person_data[::6].iterrows()):
        original_image = imread(test_row['path'])
        processed_image = show_object_location(original_image, test_row)

        current_axis.imshow(processed_image)
        current_axis.axis('off')

    figure.tight_layout()

    pyplot.ion()
    pyplot.show()
