"""Main module

This module trains a machine learning model for tracking objects on video.

"""

import matplotlib.pyplot as pyplot
# import pandas
from skimage.io import imread
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import RMSprop

from data import show_object_location, get_data_from_directory


def _init():
    # Read in a set of image data
    # person_data = get_data_from_directory('Kwon_VTD\\soccer')
    person_data = get_data_from_directory('BoBot\\Vid_J_person_floor')

    # with pandas.option_context('display.max_rows', 999):
    #     print(person_data)

    # _show_samples(person_data)

    # Create pre-trained model from weights
    local_weights_file = 'models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
    pre_trained_model = MobileNetV2(input_shape=(140, 140, 3),
        include_top=False, weights=None)
    pre_trained_model.load_weights(local_weights_file)

    # Don't train pre-trained model weights
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Get intermediate layer for input to our fully-connected layers
    last_layer = pre_trained_model.get_layer('block_13_expand_relu')
    print('last layer output shape:', last_layer.output_shape)
    last_output = last_layer.output

    # Append our fully-connected layers
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # Configure and compile the model
    pre_trained_model = Model(pre_trained_model.input, x)
    pre_trained_model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.0001),
                metrics=['acc'])

    print(pre_trained_model.summary())

    input('Press (almost) any key to exit.')

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


if __name__ == '__main__':
    _init()
