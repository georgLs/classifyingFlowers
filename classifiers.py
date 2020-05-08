# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:22:58 2019

@author: aaron
"""

import numpy as np
#from sets import Set
import matplotlib.pyplot as plt
from collections import Counter
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

class Classifier:
    """
    Attributes:
    ----------
    kb_imageData : array
        The image knowledgebase
    kb_labels : array
        The label to each image
    f : function
        Apply image transformation to the images
        
    """
    def __init__(self, trainingsImages, trainingsLabels, transform = lambda x: x):
        self.kb_imageData = np.array([transform(img) for img in trainingsImages])
        self.kb_labels = trainingsLabels
        self.f = transform

    def classifyImages(self, testImages, probs = False):
        """
        Classifies a set of images
        
        Parameters
        ----------
        testImages : array
            The images to classify
            
        Returns
        ----------
        array
            The labels assigned to each image in the order they were given
        """
        return [self.classifyImage(img, probs) for img in testImages]
        
    def classifyImage(self, img, probs = False):
        """
        Classify an image. To be defined in the sublass
        
        Parameters
        ----------
        img : ndarray
            The image you want to classify
        
        Returns
        ----------
        label
            The predicted label
        """
        return self.classifyDescriptor(self.f(img), probs)
        
    def classifyDescriptor(self, descriptor, probability_distribution):
        raise NotImplementedError("Define the classification logic")
    
    def evaluate(self, testImages, testLabels, show_failings = True):
        predictions_prob = self.classifyImages(testImages, True)
        predictions = [max(p, key = lambda x: x[1])[0] for p in predictions_prob]
        
        positiveCount = 0
        for i in range(len(predictions)):
            if predictions[i] == testLabels[i]:
                positiveCount += 1
        print(confusion_matrix(testLabels, predictions))
        
        errors = np.where(np.array(predictions) != np.array(testLabels))[0]
        print("Accuracy:", 1 - len(errors) / len(testLabels))
        
        if show_failings:
            for i in errors:
               plt.imshow(testImages[i])
               plt.show()
               predictions_prob[i].sort(key = lambda x: x[1])
               predictions_prob[i].reverse()
               print("Original label:", testLabels[i], ", Predicted label:", predictions[i], ", Confidence:",
                     predictions_prob[i][0][1], ", Top 3 guesses:", predictions_prob[i][:3])
        
    
    def stack(labels, args):
        """
        Sum of the probabilitiy distributions of multiple classifiers

        Parameters
        ----------
        args : Classifiers*
            The classifiers

        Returns
        ----------
        The probabilities summed up
        """
        prob = {label: 0 for label in labels}
        for label in labels:
            for arg in args:
                for pair in arg:
                    if pair[0] == label:
                        prob[label] += pair[1]
        return list(prob.items())
    
    def highest_confidence(labels, args):
        prob = {label: 0 for label in labels}
        for label in labels:
            for arg in args:
                for pair in arg:
                    if pair[0] == label:
                        prob[label] = max(pair[1], prob[label])
        return list(prob.items())
                        
            
    
class KNNClassifier(Classifier):
    """
    A image classifier that uses the k nearest neighbours for classification.
    
    Attributes
    ----------
    kb_imageData : array
        The descriptors to all images for the classifier to refer to
    kb_labels : array
        The labels of the images in kb_images
    descFunc : func
        A function that returns the descriptors of an image
    k : int
        The number of neighbours considered for the classification of an image
    weigths : array
        Specify the weights given to each descriptor. Every descriptor is
        weighted 1 by default.
    """
    
    def __init__(self, trainingsImages, trainingsLabels, descriptorFunction = lambda x: x, k = 1, weights = []):
        super().__init__(trainingsImages, trainingsLabels, descriptorFunction)
        self.k = k
        self.weights = weights
        if weights == None or len(weights) != len(self.kb_imageData[0]):
            self.weights = [1] * len(self.kb_imageData[0])
            print("Number of weights does not match number of descriptors, using default weighting")
            
    def distance(self, desc1, desc2):
        """
        Calculate the euclidean distance between two images using the Classifiers descriptor
        function
        
        Parameters
        ----------
        img1 : ndarray
            The descriptor of the first image
        img2 : ndarray
            The descriptor of the second image
        weights : array
            Specify a weight given to each descriptor.
            If unspecified, every descriptor is weighted 1
            
        Returns
        ----------
        distance : int
            The euclidean distance between the images
        """
        d = 0
        for i in range(len(desc1)):
            d += (desc1[i] - desc2[i]) ** 2 * self.weights[i]
        return d ** (1/2)

    def classifyDescriptor(self, img, probs):
        """
        Classify an image using the k nearest images
        
        Parameters
        ----------
        img : ndarray
            The image you want to classify
        
        Returns
        ----------
        label
            The most common label of the k-nearest images
        """
        k_nearest = [(self.distance(img, self.kb_imageData[i]),
                      self.kb_labels[i]) for i in range(len(self.kb_imageData))]
        k_nearest.sort(key = lambda t: t[0])
        k_nearest = k_nearest[:self.k]
        if probs:
            counter = Counter([x[1] for x in k_nearest])
            labelCounter = Counter(self.kb_labels)
            return [(label, counter[label] / self.k) for label in sorted(labelCounter)]
        return Counter(list(zip(*k_nearest))[1]).most_common(1)[0][0]


class NNClassifier(Classifier):
    """
    Classifiy images by the use of a neural network.
    
    Attributes
    ----------
    kb_imageData : array
        The descriptors to all images for the classifier to refer to
    kb_labels : array
        The labels of the images in kb_images
    neuralModel : keras.Sequential
        The neural network
    encoder :
    """
    
    def __init__(self, trainingsImages, trainingsLabels, neuralNetworkModel, transformFunction = lambda x: x):
        super().__init__(trainingsImages, trainingsLabels, transformFunction)
        self.neuralModel = neuralNetworkModel
        self.encoder = LabelEncoder()
        self.encoder.fit(trainingsLabels)
        self.imageGenerator = ImageDataGenerator(vertical_flip=True, featurewise_center = True,
                                                 rotation_range = 15, fill_mode = "constant", cval = .0)

    def train(self, epochs, batchSize, useGenerator = False, verbose = 0):
        """
        Train the underlying neural network.
        
        Parameters
        ----------
        epochs : int
            Number of training runs performed
        """
        if not useGenerator:
            self.neuralModel.fit(self.kb_imageData, to_categorical(self.encoder.transform(self.kb_labels)), 
                                 batch_size = batchSize, epochs = epochs, validation_split = 0.20, verbose = verbose,
                                 callbacks = [EarlyStopping(monitor="val_loss", min_delta=0, patience=3),
                                              ModelCheckpoint("./weights.h5", monitor = "val_loss", verbose = 0, save_best_only = True, 
                                                              save_weights_only = False, mode = "auto", period = 1),
                                              ReduceLROnPlateau(monitor = "val_loss", factor = .1,
                                                               patience = 3, verbose = verbose, mode = "min")])
        else:
            val_split = int(len(self.kb_imageData) * 0.8)
            labels = self.encoder.transform(self.kb_labels)
            print(val_split)
            self.neuralModel.fit_generator(self.imageGenerator.flow(self.kb_imageData[:val_split],
                                                                    to_categorical(labels[:val_split]),
                                                                    batch_size = batchSize, 
                                                                    shuffle = False), 
                validation_data = (self.kb_imageData[val_split:], to_categorical(labels[val_split:])),
                validation_steps = 1, steps_per_epoch = 4 * val_split / batchSize, epochs = epochs, verbose = verbose,
                callbacks = [EarlyStopping(monitor="val_loss", min_delta=0, patience=5),
                             ModelCheckpoint("./weights.h5", monitor = "val_loss", verbose = 0,
                                             save_best_only = True, save_weights_only = False,
                                             mode = "auto", period = 1),
                                             ReduceLROnPlateau(monitor = "val_loss", factor = .1,
                                                               patience = 2, verbose = verbose, mode = "min")])
        
        self.neuralModel.load_weights("./weights.h5", by_name = True)
        
    def classifyDescriptor(self, image, probs):
        """
        Classify an image
        
        Parameters
        ----------
        image : ndarray
            the image to classify
            
        Returns
        ----------
        class : int, float, string
            the predicted label.
            Datatype is the same used for the trainingslabels upon initialization.
        """
        if probs:
            return [(self.encoder.inverse_transform([i])[0], x) for i, x in enumerate(self.neuralModel.predict(np.array([image, ]))[0])]
        return self.encoder.inverse_transform([np.argmax(self.neuralModel.predict(np.array([image,])))])[0] 
    
class EnsembleClassifier(Classifier):
    def __init__(self, confidence_mode = "highest"):
        """
        Parameters
        ----------
        labels : set
            The labels used for classification
        """
        self.f = lambda x: x
        self.classifiers = []
        self.classifier_usedSet = []
        self.kb_labels = set()
        self.confidence = confidence_mode
        
    def add(self, classifier, used_set = 0):
        """
        Add a pre-defined classifier
        """
        self.classifiers.append(classifier)
        self.classifier_usedSet.append(used_set)
        self.kb_labels.update(classifier.kb_labels)
        
    def classifyImage(self, img, probs = False):
        """
        Use all added classifiers to predict the label for an image.

        Parameters
        ---------
        img : ndarray
            The image to predict
        mode : string
            
        probs = False
            If True, returns the distribution of all predictions over all classifiers,
            else and by dafault: returns the best prediction
        """
        prob = -1
        if self.confidence == "average":
            #prob = [(p[0], p[1] / len(self.classifiers)) for p in Classifier.highest_confidence(self.kb_labels, [c.classifyImage(img, True) for c in self.classifiers])]
            prob = [(p[0], p[1] / len(self.classifiers)) for p in Classifier.stack(self.kb_labels, [c.classifyImage(img, True) for c in self.classifiers])]
        elif self.confidence == "highest":
            prob = Classifier.highest_confidence(self.kb_labels, [c.classifyImage(img, True) for c in self.classifiers])
        if probs:
            return prob
        return max(prob, key = lambda x: x[1])[0]
    
    
    """
    Experimental functions to mix usage of segmented and unedited images
    """
    def classifyMixedImage(self, img, probs = False):
 
        prob = -1
        if self.confidence == "average":
            prob = [(p[0], p[1] / len(self.classifiers)) for p in Classifier.stack(self.kb_labels, [self.classifiers[i].classifyImage(img[self.classifier_usedSet[i]], True) for i in range(len(self.classifiers))])]
        elif self.confidence == "highest":
            prob = Classifier.highest_confidence(self.kb_labels, [self.classifiers[i].classifyImage(img[self.classifier_usedSet[i]], True) for i in range(len(self.classifiers))])
        if probs:
            return prob
        return max(prob, key = lambda x: x[1])[0]
    
    def classifyMixedImages(self, images, images_per_set, probs = False):

        return [self.classifyMixedImage(images[i::images_per_set], probs) for i in range(images_per_set)]
    
    def evaluateMixedImageSets(self, testImages, testLabels, show_failings = True):
        predictions_prob = self.classifyMixedImages(testImages, len(testLabels), True)
        predictions = [max(p, key = lambda x: x[1])[0] for p in predictions_prob]
        
        positiveCount = 0
        for i in range(len(predictions)):
            if predictions[i] == testLabels[i]:
                positiveCount += 1
        print(confusion_matrix(testLabels, predictions))
        
        errors = np.where(np.array(predictions) != np.array(testLabels))[0]
        print("Accuracy:", 1 - len(errors) / len(testLabels))
        
        if show_failings:
            for i in errors:
               plt.imshow(testImages[i])
               plt.show()
               predictions_prob[i].sort(key = lambda x: x[1])
               predictions_prob[i].reverse()
               print("Original label:", testLabels[i], ", Predicted label:", predictions[i], ", Confidence:",
                     predictions_prob[i][0][1], ", Top 3 guesses:", predictions_prob[i][:3])