"""Main module

This module trains a machine learning model for tracking objects on video.

"""

from enum import Enum
from glob import glob
import math
import os

import matplotlib.pyplot as pyplot
import numpy
import pandas
from skimage.io import imread

from data import show_object_location, get_data_by_directory


def _init():
    # Read in a set of image data
    # person_data = get_data_by_directory('Kwon_VTD\\soccer')
    person_data = get_data_by_directory('BoBot\\Vid_J_person_floor')
    person_data.dropna(inplace=True)
    person_data.sample(3)

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

    input('Press (almost) any key to exit.')


if __name__ == '__main__':
    _init()
