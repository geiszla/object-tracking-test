"""Data module

This module contains functions for data I/O and manipulation.

"""

from glob import glob
import os
import re

import numpy
from pandas import read_table


COLUMNS = ['topLeftX', 'topLeftY', 'bottomRightX', 'bottomRightY']


def get_data_from_directory(directory_name, is_append_path=False):
    """
    Gets the data-set from the first directory found, whose path includes the given string.

    Args:
        directory_name (str): The name to search directories by

    Returns:
        pandas.DataFrame: The data read in from the matching directory
    """

    # Find directory
    data_directories = sorted([directory_path for directory_path, _, file_names in os.walk('./data')
        if 'gt.txt' in file_names and directory_name in directory_path])

    # Read in data-set metadata
    file_path = os.path.join(data_directories[0], 'gt.txt')
    data = read_table(file_path, header=None, sep=r'[\s,]+', names=COLUMNS, engine='python')

    # Remove empty entries and convert location data to integers
    data.dropna(inplace=True)
    data = data.apply(lambda row: row.apply(lambda value: int(round(value))))

    # Read in image file paths and add it to the data if required
    image_files = glob(os.path.join(data_directories[0], '*.jpg'))

    if is_append_path:
        image_number_regex = re.compile(r'\d+$')
        image_paths = {
            int(image_number_regex.search(os.path.splitext(file_path)[0]).group(0)): file_path
                for file_path in image_files
        }
        data['path'] = data.index.map(lambda index: image_paths.get(index + 2))

    print('Found {} image files in {}.'.format(len(image_files), data_directories[0]))

    return data, data_directories[0]

def show_object_location(original_image, data_row):
    """
    Applies an alpha layer to the parts of the image that don't contain the object highlighting the
    part that do.

    Args:
        original_image (numpy.ndarray): The image to hightlight the object on
        data_row (pandas.Series): The data, which contains the location of the object

    Returns:
        numpy.ndarray: The image with the object hightlighted on it
    """

    alpha_layer = numpy.zeros(original_image.shape[:2]+(1,), dtype=numpy.uint8)
    processed_image = numpy.concatenate([original_image, alpha_layer], 2)

    # Add alpha layer to indicate location of object
    processed_image[:, :, 3] = 127
    processed_image[data_row['topLeftY']:data_row['bottomRightY'],
        data_row['topLeftX']:data_row['bottomRightX'], 3] = 255

    return processed_image
