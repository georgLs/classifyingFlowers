# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:00:47 2019

@author: Aaron
"""

import glob, os, scipy.io, skimage.io, skimage.color

def splitDirectory(sourceDir, targetDir, labels):
    """
    Splits the images from a source directory into seperate class directories
    within a target directory. It refers to the numbering of the file names to
    retrieve the label of an image from the 'labels' array
        E.g. The label of the image 'image_06734.jpg' has to be stored at
        index 6733 in the 'labels' array
    
    Parameters
    ----------
    sourceDir : String
        The source directory that contains the images
    targetDir : String
        The target directory the images will be moved into
    labels : array
        The labels of the images
    """
    if not os.path.exists(sourceDir):
        print("Source directory does not exist")
        return
    if not os.path.exists(targetDir):
        print("Creating target folder...")
        try:
            os.makedirs(targetDir)
        except OSError as e:
            print(e)
            return

    imageFiles = []
    for (dirpath, dirnames, filenames) in os.walk(sourceDir):
        imageFiles.extend(filenames)

    for img in imageFiles:
        i = int(img.split(".")[0].split("_")[1]) - 1
        classDir = targetDir + "/" + str(labels[i])
        if not os.path.exists(classDir):
            os.makedirs(classDir)
        print("Moving", sourceDir + "/" + img, "to", classDir + "/" + img)
        os.rename(sourceDir + "/" + img, classDir + "/" + img)
        
 # splitDirectory("C:/.../jpg", "C:/.../cjpg", getLabels())

def getLabels(labelFile="imagelabels.mat"):
    """
    Loads the labels from a matlab-file
    
    Parameters
    ----------
    labelFile : String, optional
        The .mat-file that contains the labels
        The default is "imagelabels.mat"
        
    Returns
    ----------
    The labels in an array
    """

    mat = scipy.io.loadmat(labelFile)
    return mat['labels'][0]

def loadClass(sourceDir, label):
    """
    Load all images of one class
    
    Parameters
    ----------
    sourceDir : String
        The source directory that contains the images. Images must be
        categorized into subfolders according to their classes (see splitDirectory())
    label : String_like
        The label of the class. Must be possible to cast into a String
        
    Returns
    ----------
    images : array
        The images
    """
    if not os.path.exists(sourceDir):
        print("Source directory does not exist")
        return
    
    images = []
    classDir = sourceDir + "/" + str(label)
    if os.path.exists(classDir):
        for image_path in glob.glob(classDir + "/*.jpg"):
            images.append(skimage.io.imread(image_path))
    return images

def transformDir(sourceDir, targetDir, f, classes = []):
    """
    Apply a function to all images in the source directory and save the 
    processed images to a new directory
    
    Parameters
    ----------
    sourceDir : String
        The source directory that contains the images. Images must be
        categorized into subfolders according to their classes
    targetDir : String
        The target directory the images will be moved into
    f : function
        Function that takes the image as ndarray and processes it
    classes : array
        The classes to consider
    """
    
    if not os.path.exists(sourceDir):
        print("Source directory does not exist")
        return
    
    if not os.path.exists(targetDir):
        print("Creating target folder...")
        try:
            os.makedirs(targetDir)
        except OSError as e:
            print(e)
            return

    for c in classes:
        classDir = sourceDir + "/" + str(c)
        if os.path.exists(classDir):
            imageFiles = []
            for (dirpath, dirnames, filenames) in os.walk(classDir):
                imageFiles.extend(filenames)
            
            for file in imageFiles:
                imPath = targetDir + "/" + str(c)
                if not os.path.exists(imPath):
                    os.makedirs(imPath)
                    
                imPath += "/" + file
                image = skimage.io.imread(classDir + "/" + file)
                image = f(image)
                skimage.io.imsave(targetDir + "/" + str(c) + "/" + file, image)
                print("Saved file", str(c) + "/" + file)
    
    
def loadCategorizedImages(sourceDir, classes = [], color_space = "RGB"):
    """
    Load images
    
    Parameters
    ----------
    sourceDir : String
        The source directory that contains the images. Images must be
        categorized into subfolders according to their classes (see splitDirectory())
    classes : array
        n array that specifies the classes you want to use.
        E.g. classes=[0, 1] to load classes 0 and 1
        Loads all classes by default.
    color_space : String, optional
        "RGB" or "HSV" to load the images in the specified color format
        
    Returns
    ---------
    imgs : array
        The images
    labels : array
        The label of the images
    """

    if not os.path.exists(sourceDir):
        print("Source directory does not exist")
        return
    images = []
    labels = []
    if len(classes) == 0:
        for (_, dirnames, _) in os.walk(sourceDir):
            classes.extend(dirnames)
    for c in classes:
        classDir = sourceDir + "/" + str(c)
        if os.path.exists(classDir):
            if color_space == "RGB":
                for image_path in glob.glob(classDir + "/*.jpg"):
                    images.append(skimage.io.imread(image_path))
                    labels.append(c)
            elif color_space == "HSV":
                for image_path in glob.glob(classDir + "/*.jpg"):
                    images.append(skimage.color.rgb2hsv(skimage.io.imread(image_path)))
                    labels.append(c)

    return images, labels

    
