# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:00:16 2019

@author: aaron
"""

import utils.loading, utils.data, classifiers, meta
from imagetools import analysis, processing, processingO

import numpy, skimage
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.feature import hog
from sklearn.metrics import confusion_matrix

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import VGG16, VGG19
from keras.applications.mobilenet import MobileNet
from keras import backend as kb
from keras.backend.tensorflow_backend import get_session, set_session, clear_session
from keras.preprocessing.image import ImageDataGenerator

sess = get_session()
clear_session()
sess.close()
sess = get_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
kb.tensorflow_backend.set_session(tf.Session(config = config))

#classes = meta.classes_by_size[:3, 0]
classes =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
imagesize = (224, 224, 3)


def loadOriginalImages(classes = classes, imagesize = imagesize):
    dC = utils.data.DataContainer()
    dC.insertData(*utils.loading.loadCategorizedImages("./_res/cjpg", classes, "RGB"))
    return dC.pullDataSplit(test_size = 5, shuffle = False)

def loadSegmentedImages(classes = classes, imagesize = imagesize):
    dC = utils.data.DataContainer()
    dC.insertData(*utils.loading.loadCategorizedImages("./_res/sjpg", classes, "RGB"))
    return dC.pullDataSplit(test_size = 5, shuffle = False)

tr_images_O, tr_labels_O, test_images_O, test_labels_O = loadOriginalImages() #The original, unedited images
tr_images_S, tr_labels_S, test_images_S, test_labels_S = loadSegmentedImages()
#tr_images, tr_labels, val_images, val_labels = utils.data.validationSplit(tr_images, tr_labels, .2)

def mean(imagesize = imagesize):
    knn = classifiers.KNNClassifier(tr_images_O, tr_labels_O, lambda x: analysis.mean(x), k=10)
    return knn

def hist():
    knn = classifiers.KNNClassifier(tr_images_O, tr_labels_O, lambda x: analysis.histogram(x), k=10)
    return knn

def cnnTest():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = "relu", padding = "same", input_shape = imagesize))
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(128, (3, 3), activation = "relu", padding = "same"))
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(256, (3, 3), activation = "relu"))
    model.add(Conv2D(256, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(512, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation = "relu", input_shape = (32, )))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation = "softmax"))
    model.summary()
    
    trainModel(model)
    
def hogNearestNeighbour():
    
    knn = classifiers.KNNClassifier(tr_images_O, tr_labels_O, lambda x: hog(resize(x, imagesize, mode = "constant", anti_aliasing=True),
                                                                        block_norm = "L2-Hys", orientations = 8,
                                                                        pixels_per_cell = (14, 14), cells_per_block = (1, 1), 
                                                                        visualize = False, multichannel = True), k=5)
    
    return knn
    
def hogNN(image_size = imagesize, cell_size = (14, 14), orientations = 8):
    model = Sequential()
    #model.add(Dense(1024, activation = "relu", input_shape = (2048,)))
    model.add(Dense(1024, activation = "relu", input_shape = ( int(numpy.ceil(image_size[0] / cell_size[0]) * numpy.ceil(image_size[1] / cell_size[1]) * orientations) ,)))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(len(classes), activation = "softmax"))
    
    model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.001, momentum = 0.9), metrics=["accuracy"])
                
    nn = classifiers.NNClassifier(tr_images_O, numpy.array(tr_labels_O), model, lambda x: hog(resize(x, image_size,
                                  mode = "constant", anti_aliasing = True), block_norm = "L2-Hys", orientations = orientations, 
        pixels_per_cell = cell_size, cells_per_block = (1, 1), visualize = False, multichannel = True))
    nn.train(500, 10)
    return nn

def vgg16(classes = classes, imagesize = imagesize):
    vgg16 = VGG16(weights = "imagenet", include_top = False, input_shape = imagesize)

    for layer in vgg16.layers[:-4]:
        layer.trainable = False
        
    for layer in vgg16.layers:
        print(layer, layer.trainable)
                
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(.5))
    model.add(Dense(len(classes), activation = "softmax"))
    model.summary()
    return model
    
def vgg19(classes = classes, imagesize = imagesize):
    vgg19 = VGG19(weights = "imagenet", include_top = False, input_shape = imagesize)
        
    for layer in vgg19.layers[:-4]:
        layer.trainable = False
        
    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(.2))
    model.add(Dense(len(classes), activation = "softmax"))
    model.summary()
    return model
    
def mobilenet(classes = classes, imagesize = imagesize):
    mobilenet = MobileNet(include_top = False, input_shape = imagesize)

    for layer in mobilenet.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(mobilenet)
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(.5))
    model.add(Dense(len(classes), activation = "softmax"))
    model.summary()
    return model
    
def trainModel(model, imagesize = imagesize, useGenerator = False):
    model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.001, momentum = 0.9), metrics=["accuracy"])
    
    nn = classifiers.NNClassifier(tr_images_O, numpy.array(tr_labels_O), model, lambda x: resize(x, imagesize, mode = "constant", anti_aliasing = True))
    nn.train(200, 5, useGenerator, verbose = True)
    
    return nn

def loadModelWeights(model, file, imagesize = imagesize):
    model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.001, momentum = 0.9), metrics=["accuracy"])
    model.load_weights(file, by_name = True)
    
    return classifiers.NNClassifier(tr_images_O, numpy.array(tr_labels_O), model, lambda x: resize(x, imagesize, mode = "constant", anti_aliasing = True))

def testEnsamble():
    v16 = trainModel(vgg16())
    print("VGG-16:")
    v16.evaluate(test_images_O, test_labels_O)
    v19 = trainModel(vgg19())
    print("VGG-19:")
    v19.evaluate(test_images_O, test_labels_O)
    #mn = trainModel(mobilenet())
    #print("mobilenet:")
    #mn.evaluate(test_images_O, test_labels_O)
    mc = classifiers.EnsembleClassifier()
    mc.add(v16)
    mc.add(v19)
    #mc.add(mn)
    print("VGG-16 + VGG-19 + mobilenet:")
    mc.evaluate(test_images_O, test_labels_O)
    
def testClassic():
    m = classifiers.KNNClassifier(tr_images_S, tr_labels_O, lambda x: analysis.mean(x), k=10)
    m.evaluate(test_images_S, test_labels_O, False)
    
    h = classifiers.KNNClassifier(tr_images_O, tr_labels_O, lambda x: analysis.histogram(x), k=10)
    h.evaluate(test_images_O, test_labels_O, False)
    
    hnn = hogNN()
    hnn.evaluate(test_images_O, test_labels_O, False)
    
    mc = classifiers.EnsembleClassifier(confidence_mode = "average")
    mc.add(hnn, 0)
    mc.add(m, 1)
    mc.evaluateMixedImageSets(test_images_O + test_images_S, test_labels_O, False)
    mc.add(hist(), 0)
    mc.evaluateMixedImageSets(test_images_O + test_images_S, test_labels_O, False)
    
def testAll():
    m = mean()
    m.evaluate(test_images_O, test_labels_O, False)
    h = hogNearestNeighbour()
    h.evaluate(test_images_O, test_labels_O, False)
    hnn = hogNN()
    hnn.evaluate(test_images_O, test_labels_O, False)
    vgg = trainModel(vgg16(), useGenerator = True)
    vgg.evaluate(test_images_O, test_labels_O)
    mc = classifiers.EnsembleClassifier()
    mc.add(vgg)
    mc.add(hnn)
    mc.add(m)
    mc.evaluate(test_images_O, test_labels_O)
