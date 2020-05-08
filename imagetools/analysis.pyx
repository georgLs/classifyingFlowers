# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:58:23 2019

@author: Aaron
"""

import cython
import numpy as np
cimport numpy as np


def mean(np.ndarray image):
    """
    The means per color channel of an image.
	Ignores 0.
    
    Parameters
    ----------
    
    image : ndarray
        The image as numpy array

    Returns
    ..........
    (r, g, b) : (int, int, int)
        A of the means per color channel
    """
    if image.dtype == np.uint8:
        return mean_uint8(image)
    if image.dtype == np.float_:
        return mean_float(image)
    else:
        print("Datatype not supported")
        
def histogram(np.ndarray image, bins = 8):
    if image.dtype == np.uint8:
        r = (0, 256)
    elif image.dtype == np.float_:
        r = (0.0, 1.0)
    return np.hstack((np.histogram(image[:, :, 0], bins = bins, range = r)[0],
                     np.histogram(image[:, :, 1], bins = bins, range = r)[0],
                     np.histogram(image[:, :, 2], bins = bins, range = r)[0]))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef mean_float(np.ndarray[np.float_t, ndim = 3] image):
    cdef float[3] a = [0, 0, 0]
    cdef int i, x, y
    cdef int size
    cdef float value
    for i in range(3):
        size = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                value = image[x, y, i]
                if value > 0:
                    a[i] += value
                    size += 1
        a[i] = a[i] / size
    return a[0], a[1], a[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mean_uint8(np.ndarray[np.uint8_t, ndim=3] image):
    cdef float[3] a = [0, 0, 0]
    cdef int i, x, y
    cdef int size
    cdef int value
    for i in range(3):
        size = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                value = image[x, y, i]
                if value > 0:
                    a[i] += value
                    size += 1
        a[i] = a[i] / size
    return a[0], a[1], a[2]