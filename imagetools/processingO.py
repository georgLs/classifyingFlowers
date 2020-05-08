# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:51:41 2019

@author: Aaron
"""
import numpy as np
import skimage.filters, skimage.measure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def binaryOtsu(image):
    """
    Turns an colored image into a monochrome binary image. Uses the
    ostu-treshold filter
    
    Parameters
    ----------
    image : ndarray
        The image
    
    Returns
    ----------
    binary : ndarray
        The binary image
    """
    greyImg = image[:, :, 0] / 3 + image[:, :, 1] / 3 + image[:, :, 2] / 3
    threshold = skimage.filters.threshold_otsu(greyImg)
    return greyImg > threshold

"""
def binaryC(image):
    #c1, c2 = clustering(image, image[0, 0, :], image[image.shape[0] // 2, image.shape[1] // 2, :])
    bg = np.array([0.0, 0.0, 0.0])
    edge = image.shape[0] * 2 + image.shape[1] * 2 - 4
    for x in range(-1, 1):
        for y in range(image.shape[1]):
            bg += image[x, y, :]
    for y in range(-1, 1):
        for x in range(1, image.shape[0] - 1):
            bg += image[x, y, :]
    bg = bg / edge
    fg = [180, 30, 120]#np.mean(image) + (np.mean(image) - bg)

    c1, c2 = clustering(image, k=2, clusters=np.array([bg, fg]))
    print(c1, c2)
    binary = np.copy(image[:, :, 0])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            s = (1 if (c1[0] + c1[2]) / c1[1] < (c2[0] + c2[2]) / c2[1] else -1)
            if s * dist(image[x, y, :], c1) < s * dist(image[x, y, :], c2):
                binary[x, y] = 0
            else:
                binary[x, y] = 1
    return binary
"""

def edge(image):
    bg = np.array([0.0, 0.0, 0.0])
    for x in range(-1, 1):
        for y in range(image.shape[1]):
            bg += image[x, y, :]
    for y in range(-1, 1):
        for x in range(1, image.shape[0] - 1):
            bg += image[x, y, :]
    edge = image.shape[0] * 2 + image.shape[1] * 2 - 4
    bg = bg / edge
    return bg

def binarySK(image, verbose = True):
    
    best = 0
    clusters = 2

    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(image.reshape(image.size // 3, 3))

    foreground_cluster = None
    best = 0

    for i, cluster in enumerate(kmeans.cluster_centers_):
        #cluster = [(cluster[0] - bg[0]) ** 2, cluster[1] ** 2, (cluster[2] - bg[2]) ** 2]
        #RBvG_ratio = (cluster[0] + cluster[2]) / cluster[1] # ratio of red and blue colors vs green color
        RBvG_ratio = max(cluster[0] / cluster[1], cluster[2] / cluster[1]) # max of ratios red/green and blue/green
        if RBvG_ratio > best:
            best = RBvG_ratio
            foreground_cluster = i
        if verbose:
            print("Cluster", i, cluster, "RBvG:", RBvG_ratio)
        
    if verbose:
        print("Detected", kmeans.cluster_centers_[foreground_cluster], "as foreground")
    
    binary = np.zeros((image.shape[0], image.shape[1]), dtype = bool)
    for x in range(binary.shape[0]):
        for y in range(binary.shape[1]):
            closest = np.argmin(np.array([dist(image[x, y, :], c) for c in kmeans.cluster_centers_]))
            if closest == foreground_cluster:
                binary[x, y] = True
            else:
                binary[x, y] = False
                
    return binary

def binarySK_similarity_score(image, k=3, verbose = True):
    from skimage.transform import resize
    
    sil_img = resize(image, (150, 150, 3), mode = "constant", anti_aliasing = False) # Downscale the image for calculating the silhouette score
    sil_img = sil_img.reshape((sil_img.size // 3, 3)) # silhouette_score takes 2d inputs
    
    best = 0
    clusters = 1
    for i in range(2, k+1):
        kmeans = KMeans(n_clusters = i, random_state = 0).fit(sil_img)
        sil_score = silhouette_score(sil_img, kmeans.labels_)
        #sil_score = davies_bouldin_score(sil_img, kmeans.labels_)
        if verbose:
            print("Silhouette score with", i, "clusters:", sil_score)
        if sil_score > best:
            best = sil_score
            clusters = i

    if verbose:
        print("Using", clusters, "clusters")
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(image.reshape(image.size // 3, 3))

    foreground_cluster = None
    best = 0
    bg = edge(image)
    for i, cluster in enumerate(kmeans.cluster_centers_):
        #cluster = [(cluster[0] - bg[0]) ** 2, cluster[1] ** 2, (cluster[2] - bg[2]) ** 2]
        #RBvG_ratio = (cluster[0] + cluster[2]) / cluster[1] # ratio of red and blue colors vs green color
        RBvG_ratio = max(cluster[0] / cluster[1], cluster[2] / cluster[1]) # max of ratios red/green and blue/green
        if RBvG_ratio > best:
            best = RBvG_ratio
            foreground_cluster = i
        if verbose:
            print("Cluster", i, cluster, "RBvG:", RBvG_ratio)
        
    if verbose:
        print("Detected", kmeans.cluster_centers_[foreground_cluster], "as foreground")
    
    binary = np.zeros((image.shape[0], image.shape[1]), dtype = bool)
    for x in range(binary.shape[0]):
        for y in range(binary.shape[1]):
            closest = np.argmin(np.array([dist(image[x, y, :], c) for c in kmeans.cluster_centers_]))
            if closest == foreground_cluster:
                binary[x, y] = True
            else:
                binary[x, y] = False
                
    return binary

def binary4(image):
    import meta
    binary = np.copy(image[:, :, 0])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if min([dist(i, image[x, y, :]) for i in meta.foreground_colors_rgb]) < min([dist(i, image[x, y, :]) for i in meta.background_colors_rgb]):
                binary[x, y] = 1
            else:
                binary[x, y] = 0
    return binary

def clustering(image, k=2, clusters = np.array([[.0, .0, .0], [.0, .0, .0]])):
    cl = np.array([[.0, .0, .0]] * k)
    cl_sizes = [0] * k
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            d = [dist(image[x, y, :], c) for c in clusters]
            cl[d.index(min(d))] += image[x, y, :]
            cl_sizes[d.index(min(d))] += 1
            
    for i, c in enumerate(cl):
        cl[i] = c / cl_sizes[i]

    if not np.array_equal(cl, clusters):
        return clustering(image, k=k, clusters=cl)
    return cl


def dist(a, b):
    d = 0
    for i in range(len(a)):
        d += (a[i] - b[i]) ** 2
    return d ** (1/2)

def seg(image):
    b = binarySK(image)
    return np.stack([image[:, :, 0] * b, image[:, :, 1] * b, image[:, :, 2] * b], axis=2)

def bbox(image, removeBackground = True):
    """
    Detects the main object in an image by turning the image monochrome and
    searching for the biggest region in the monochrome image. This region is
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
    labelImage = skimage.measure.label(binarySK(image))
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