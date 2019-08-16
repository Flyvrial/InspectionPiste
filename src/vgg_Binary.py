from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import keras
import cv2
import numpy as np
import random as rd
import os
import datetime
import time
from keras.utils import np_utils

###################################################################
# Paper source :
# Very Deep Convolutional Networks for Large-Scale Image Recognition
# K. Simonyan, A. Zisserman
# arXiv:1409.1556
# Keras code source :
# https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
###################################################################

class VGG_Binary:

	##
	# __init__ :
	# 	input :
	#		weights_path : the path to the trained weight, none if there is no weight
	#	Descrtiption : initialise the model with sgd optimizer
	## 
	def __init__(self,vgg,weights_path=None ):
		if vgg == 19:
			self.model = self.VGG_19(weights_path)
		else:
			self.model = self.VGG_16(weights_path)
		rmsprop = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=1e-6)
		sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

	##
	# VGG_19 :
	# 	input :
	#		weights_path : the path to the trained weight
	#	Descrtiption : load the vgg neural network from keras database and change the fully connected layer
	## 
	def VGG_19(self,weights_path):


		model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='max', classes=1000)
		for layer in model.layers:
			layer.trainable = False

		x = model.output
		#x = Flatten()(x)

		x = Dense(4096, activation='relu')(x) 
		x = Dropout(0.5)(x)
		x = Dense(4096, activation='relu')(x) 
		x = Dropout(0.5)(x)

	    # New softmax layer
		predictions = Dense(2, activation='softmax')(x) 

		finetune_model = Model(inputs=model.input, outputs=predictions)
		
		finetune_model.load_weights(weights_path)
		
		return finetune_model

	##
	# VGG_16 :
	# 	input :
	#		weights_path : the path to the trained weight
	#	Descrtiption : load the vgg neural network from keras database and change the fully connected layer
	## 
	def VGG_16(self, weights_path):
		model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='max', classes=1000)
		for layer in model.layers:
			layer.trainable = False

		x = model.output
		#x = Flatten()(x)

		x = Dense(4096, activation='relu')(x) 
		x = Dropout(0.5)(x)
		x = Dense(4096, activation='relu')(x) 
		x = Dropout(0.5)(x)

	    # New softmax layer
		predictions = Dense(2, activation='softmax')(x) 

		finetune_model = Model(inputs=model.input, outputs=predictions)
		
		finetune_model.load_weights(weights_path)
		
		return finetune_model
		
	   
