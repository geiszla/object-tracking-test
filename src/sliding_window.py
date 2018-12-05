""" Sliding Window Detection Module

This module contains the class SlidingWindow, which contains methods to detect an object on an image
using sliding windows.
"""

from time import sleep

import cv2
import numpy
from skimage.measure import compare_ssim

class SlidingWindowDetector:
    """ Sliding Window Detector Class

    This class contains methods to detect an object on an image using sliding windows.

    Args:
        positive_example (numpy.ndarray): An image which contains the object to detect.
        negative_example (numpy.ndarray): An image which does NOT contain the object to detect.

    Note: Images are in shape: (?, ?, 3)
    """

    def __init__(self, positive_example, negative_example):
        self.__positive_example = positive_example
        self.__negative_example = negative_example

        self.__true_positives = 0
        self.__false_positives = 0
        self.__true_negatives = 0
        self.__false_negatives = 0

        self.__positive_similarities = []
        self.__negative_similarities = []

    # Public methods
    def detect_object(self, image, model):
        """
        Detects an object with sliding windows on an image using a given model for object detection.

        Args:
            image (numpy.ndarray): The image (with shape (?, ?, 3)) to detect the object on.
            model (keras.models.Model): The keras model, which is used to detect the object in a
                given window on the given image.
        """

        self.__reset_statistics()

        window = SlidingWindow(image, (45, 45), 15)
        for _ in window.slider():
            # Expand the dimensions of the image to meet the dimensions of the trained model
            test_image = numpy.expand_dims(numpy.expand_dims(window.get_window(), axis=0), axis=0)
            classes = model.predict(test_image)[0]

            #Applying threshold
            if classes[0] > 0.75:
                print('Human detected with a probability: {}\tx: {}\ty: {}'
                    .format(classes[0], window.start_x, window.start_y))

                similarity = compare_ssim(window, self.__positive_example)
                self.__positive_similarities.append(similarity)

                if similarity > 0.5:
                    self.__true_positives += 1
                else:
                    self.__false_positives += 1
            else:
                print('Background with a probability: {}\tx: {}\ty: {}'
                    .format(1-classes[1], window.start_x, window.start_y))

                similarity = compare_ssim(window, self.__negative_example)
                self.__negative_similarities.append(similarity)

                if similarity > 0.5:
                    self.__true_negatives += 1
                else:
                    self.__false_negatives += 1

            self.__show_window(image, window)

    def get_statistics(self):
        """
        Returns the recall and F1 score of the previous object detection.
        """

        recall = len(self.__true_positives) / (len(self.__true_positives)
            + len(self.__false_negatives))
        f1_score = (2 * len(self.__true_positives)) / (len(self.__false_positives)
            + len(self.__false_negatives) + 2 * len(self.__true_positives))

        return (recall, f1_score)

    # Private methods
    def __reset_statistics(self):
        self.__true_positives = 0
        self.__false_positives = 0
        self.__true_negatives = 0
        self.__false_negatives = 0

        self.__positive_similarities = []
        self.__negative_similarities = []

    @staticmethod
    def __show_window(image, window):
        clone = image.copy()

        window_end = (window.start_x + window.shape[0], window.start_y + window.shape[0])
        cv2.rectangle(clone, (window.start_x, window.start_y), window_end, (0, 255, 0), 2)

        cv2.imshow("Window", clone)
        cv2.imshow("sliding_window", window)
        cv2.waitKey(1)
        sleep(0.25)


class SlidingWindow:
    """ Sliding Window Class

    Represents a sliding window and holds its properties.

    Args:
        image (numpy.ndarray): The image (with shape (?, ?, 3)) on which the window is placed
        window_size (Tuple[int]): The height and width of the window in pixels
        step_size (int): The stride with which the window is sliding on one iteration
    """

    def __init__(self, image, window_size, step_size):
        self.__image = image
        self.__step_size = step_size

        self.height = window_size[0]
        self.width = window_size[1]

        self.start_x = 0
        self.start_y = 0

    # Public methods
    def slider(self):
        """
        To be used as an iterator to slide the window by one on evey iteration.
        """

        for window_x in range(0, self.__image.shape[0], self.__step_size):
            for window_y in range(0, self.__image.shape[1], self.__step_size):
                window = self.__get_window_image(window_x, window_y)

                if window.shape[0] != self.height or window.shape[1] != self.width:
                    continue

                self.start_x = window_x
                self.start_y = window_y

                yield

    def get_window(self):
        """
        Gets the current window.

        Returns:
            numpy.ndarray: The part of the image, which is currently in the window.
        """

        return self.__get_window_image(self.start_x, self.start_y)

    # Private methods
    def __get_window_image(self, start_x, start_y):
        return self.__image[start_x:start_x + self.height, start_y:start_y + self.width]
