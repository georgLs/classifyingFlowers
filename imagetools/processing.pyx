
import cython
import numpy as np
cimport numpy as np
import skimage.filters, skimage.measure

from libc.math cimport sqrt


def binary(image):
    if image.dtype == np.uint8:
        image = image.astype(np.float_)

    cdef np.ndarray[np.float_t, ndim = 1] bg = np.array([0.0, 0.0, 0.0])
    cdef int edge = image.shape[0] * 2 + image.shape[1] * 2 - 4
    
    cdef int x, y
    for x in range(-1, 1):
        for y in range(image.shape[1]):
            bg += image[x, y, :]
    for y in range(-1, 1):
        for x in range(1, image.shape[0] - 1):
            bg += image[x, y, :]
    
    bg = bg / edge
    #cdef np.ndarray[np.float_t, ndim = 1] fg = np.mean(image) + (np.mean(image) - bg)
    cdef np.ndarray[np.float_t, ndim = 1] fg = np.mean(image, axis = (0, 1))
    
    cdef float[3] c1 = bg #image[0, 0, :]
    cdef float[3] c2 = fg #image[image.shape[0] // 2, image.shape[1] // 2, :]
    c = clustering(image.astype(np.float), c1, c2)
    c1 = c[0]
    c2 = c[1]
    cdef np.ndarray binary = np.zeros_like(image[:, :, 0], dtype = np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if dist(image[x, y, :], c1) < dist(image[x, y, :], c2):
                binary[x, y] = 0
            else:
                binary[x, y] = 1
    return binary

cdef clustering(np.ndarray[np.float_t, ndim = 3] image, float[3] cluster1, float[3] cluster2):
    cdef float[3] c1 = [0, 0, 0]
    cdef float[3] c2 = [0, 0, 0]
    cdef int c1_len, c2_len = 0
    cdef int x, y
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if dist(image[x, y, :], cluster1) < dist(image[x, y, :], cluster2):
                c1[0] += image[x, y, 0] 
                c1[1] += image[x, y, 1]
                c1[2] += image[x, y, 2]
                c1_len += 1
            else:
                c2[0] += image[x, y, 0]
                c2[1] += image[x, y, 1]
                c2[2] += image[x, y, 2]
                c2_len += 1

    pixels = image.size
    c1[0] /= c1_len
    c1[1] /= c1_len
    c1[2] /= c1_len

    c2[0] /= c2_len
    c2[1] /= c2_len
    c2[2] /= c2_len

    if c1[0] != cluster1[0] or c1[1] != cluster1[1] or c1[2] != cluster1[2] or c2[0] != cluster2[0] or c2[1] != cluster2[1] or c2[1] != cluster2[1]:
        return clustering(image, c1, c2)
    return c1, c2


cdef float dist(np.ndarray[np.float_t, ndim = 1] a, float[3] b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def seg(image):
    b = binary(image)
    return np.stack([image[:, :, 0] * b, image[:, :, 1] * b, image[:, :, 2] * b], axis=2)

def bbox(image, removeBackground = True):
    """
    Detects the main object in an image by turning the image monochrome and
    searching for the largest region in the monochrome image. This region is
    then put into a bounding box.
    
    Parameters
    ----------
    image : ndarray
        The original image
    removeBackground : boolean
        Turn all pixels black that have not been detected as the main object.
    
    Returns
    ----------
    bbox : ndarray
        The section of the original image that contains the object
    """
    labelImage = skimage.measure.label(binary(image))
    regions = skimage.measure.regionprops(labelImage)
    maxSize = 0
    maxRegion = None
    for region in regions:
        if region.area > maxSize:
            maxSize = region.area
            maxRegion = region
    minRow, minCol, maxRow, maxCol = maxRegion.bbox
    if removeBackground:
        b = maxRegion.filled_image
        return np.stack([image[minRow:maxRow, minCol:maxCol, 0] * b, 
                         image[minRow:maxRow, minCol:maxCol, 1] * b, 
                         image[minRow:maxRow, minCol:maxCol, 2] * b], axis=2)
    return image[minRow:maxRow, minCol:maxCol, :]