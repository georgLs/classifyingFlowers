# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:53:59 2019

@author: Aaron
"""

import random

def validationSplit(images, labels, validation_size = 0.2):
    """
    Splits the images into trainings and validation images.
    The validation images are taken from the front of the array.
    
    Parameters
    ----------
    images : array
        The images
    labels : array
        The labels
    validation_size : float or int
        The size of the validation data. Defaults to 0.2
        
    Returns
    ----------
    trainings_images : array
        The images for training
    trainings_labels : array
        The labels of the trainings images
    validation_images : array
        The images for validation
    validation_labels : array
        The labels of the validation images
    """
    split = (int(validation_size * len(images)) if validation_size < 1 else validation_size)
    return images[split:], labels[split:], images[:split], labels[:split]

class DataContainer:
    """
    Container for the image data.
    
    Attributes
    ----------
    classes : array
        Contains the labels of all classes that have images loaded into the container
    classSizes : array
        Stores the size of each class
    images : array
        Stores subarrays with the images of each class.
    """
    
    def __init__(self):
        self.classes = []
        self.sizes = []
        self.images = []
        
    def containsClass(self, label):
        """
        Check if a class is already present with images in the data container
        
        Parameters
        ----------
        label
            The label of the class
            
        Returns
        ----------
        containsClass : boolean
            Whether the class is present in the container
        """
        return label in self.classes
    
    def deleteClass(self, label):
        """
        Delete a class from the contaner
        
        Parameters
        ----------
        label
            The label of the class
        """
        try:
            index = self.classes.index(label)
            self.classes.pop(index)
            self.sizes.pop(index)
            self.images.pop(index)
        except ValueError:
            print("Class", label, "is not in this container")

    def insertData(self, images, labels):
        """
        Insert images into the container
        
        Parameters
        ----------
        images : array
            The images
        labels : array
            The label of each image
        """
        for i, image in enumerate(images):
            if self.containsClass(labels[i]):
                index = self.classes.index(labels[i])
                self.images[index].append(image)
                self.sizes[index] += 1
            else:
                self.classes.append(labels[i])
                self.images.append([image])
                self.sizes.append(1)
    
    def pullClass(self, label):
        """
        Receive all images of one class
        
        Parameters
        ----------
        labels
            The class 
            
        Returns
        ----------
        class_images : array
            all images of the class
        """
        return self.images[self.classes.index(label)]
            
        
    def pullDataSplit(self, classes = [], test_size = 5, balance = True, class_order = "cyclical", shuffle = False):
        """
        Provides a split of the data into trainings and test data.
        
        Parameters
        ----------
        classes : array
            The classes to load
        testSize : int, float
            The number or percentage of images per class in the test set.
            Percentage is calculated on the smallest class.
        balance : boolean
            If True, every class is evenly represent in the trainings data.
            This does not affect the distribution of classes in the test data,
            which is always even
        class_order : string
            Specify the way the images will be ordered.
            For example, given the classes = [1, 2] and class_order = "cyclical"
            the images will be returned in order [1, 2, 1, 2, 1, 2, ...], with
            class_order = "grouped" the order will be [1, 1, ..., 1, 2, 2, ..., 2].
        shuffle : boolean
            If True, the images of a class are shuffled before being split
        
        Returns
        ----------
        trImgs : array
            The images used for training
        trLabels : array
            The labels of the trainings images
        testImgs : array
            The images used for testing
        testLabels : array
            The labels of the test images
        """
        trImgs = []
        trLabels = []
        testImgs = []
        testLabels = []

        if len(classes) == 0:
            classes = self.classes

        classIndicies = [self.classes.index(x) for x in classes]
        classImages = [self.images[x] for x in classIndicies]
        cycles = min([self.sizes[x] for x in classIndicies]) if balance else max([self.sizes[x] for x in classIndicies])
        
        if shuffle:
            for c in classImages:
                random.shuffle(c)
                
        if 0 < test_size < 1:
            test_size = int(min([self.sizes[x] for x in classIndicies]) * test_size)

        if class_order == "cyclical":
            for cycle in range(cycles):
                for classIndex in classIndicies:
                    if cycle < self.sizes[classIndex]:
                        if cycle < test_size:                                   # The test images are taken from the first images of each class
                            testImgs.append(classImages[classIndex][cycle])
                            testLabels.append(self.classes[classIndex])
                        else:
                            trImgs.append(classImages[classIndex][cycle])
                            trLabels.append(self.classes[classIndex])
            
        elif class_order == "grouped":
            for classIndex in classIndicies:
                trImgs.extend(classImages[classIndex][test_size:cycles])
                testImgs.extend(classImages[classIndex][:test_size])
                
                trLabels.extend([self.classes[classIndex] for x in range(min(len(classImages[classIndex]), cycles) - test_size)])
                testLabels.extend([self.classes[classIndex] for x in range(test_size)])

        return trImgs, trLabels, testImgs, testLabels