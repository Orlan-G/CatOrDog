# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:17:03 2021

@author: orlan

#check the dataset here https://www.kaggle.com/chetankv/dogs-cats-images
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class CNN():
    
    def __init__(self,image_input_shape):
        """
            Create the model
        """
        self.input_shape = image_input_shape
        self.classifier = tf.keras.models.Sequential() #Linear Stack of layer
        self.conv_layers(image_input_shape) #add convolutional and pooling layers
        self.dense_layers() #add "neural network" layers
        self.classifier.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    
        
    def dense_layers(self):
        """
            Initializes all "Neural" layers.
        """
        self.classifier.add(tf.keras.layers.Flatten()) #Flattening of the conv output
        self.classifier.add(tf.keras.layers.Dense(128, activation='relu')) #layer(s) of the neural network
        self.classifier.add(tf.keras.layers.Dense(1, activation='sigmoid')) #output layer (last hiden layer)
    
    def conv_layers(self, input_shape):
        """
            Initializes all layers for convolutional steps.
            There is 3 Convolution layers and 2 pooling layers 
        """
        self.classifier.add(tf.keras.layers.Conv2D(filters=32, 
                                                   kernel_size=3, 
                                                   activation='relu', 
                                                   input_shape=input_shape))
        
        self.classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, 
                                                         strides=(2,2)))
        
        self.classifier.add(tf.keras.layers.Conv2D(filters=32,
                                                   kernel_size=3,
                                                   activation='relu'))
        
        self.classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2,
                                                         strides=(2,2)))
        
        self.classifier.add(tf.keras.layers.Conv2D(filters=64,
                                                   kernel_size=3,
                                                   activation='relu'))
        
        
    def train(self, training_set, test_set):
        """
            Model training with 20 epochs
        """
        self.classifier.fit_generator(training_set,
                                      epochs = 20,
                                      validation_data = test_set)
    
    def save(self, filepath):
        """
            Save the model configuration in filepath
        """
        self.classifier.save(filepath)
        
    def load(self, filepath):
        """
            Load the model configuration in filepath
        """
        self.classifier = tf.keras.models.load_model(filepath)
        
    def accuracy_on_data(self, eval_dataset):
        """
            Evaluate the model using eval_dataset and give the accuracy
        """
        loss, acc = self.classifier.evaluate_generator(generator=eval_dataset, steps=len(eval_dataset), verbose=0)
        print('Accuracy on the dataset : ' + str(acc * 100.0))
        return loss, acc
    
    
    def prediction(self, filepath):
        """
            Make a prediction from an image(filepath)
        """
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=self.input_shape)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        value = self.classifier.predict(image)
        
        if value == 1:
            print("it is a dog :)")
        else:
            print("it is a cat :}")
        
    def model_struct(self):
        """
            Show model structure
        """
        self.classifier.summary()

def create_train_and_test_sets(training_path, test_path, target_size, batch_size, class_mode):
    """
        Create the training and test sets from training_path and test_path.
        Resize images to the target_size to fit with the model convolutional input.
    """
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(training_path,
                                                 target_size = target_size,
                                                 batch_size = batch_size,
                                                 class_mode = class_mode)

    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size = target_size,
                                                batch_size = batch_size,
                                                class_mode = class_mode)
    
    return training_set, test_set
    
if __name__ == '__main__':
    
    training_set, test_set = create_train_and_test_sets(training_path='dataset/training_set',
                                                        test_path='dataset/test_set',
                                                        target_size=(64,64),
                                                        batch_size=32,
                                                        class_mode='binary')
    
    cnn = CNN(image_input_shape=(64,64,3))
    cnn.model_struct()
    
    cnn.train(training_set, test_set) #train the model
    
    
    #cnn.load('ExampleCatOrDogModel') #To run an example with 82% accuracy
    
    loss, acc = cnn.accuracy_on_data(test_set) #evaluate the model   
    
    #cnn.prediction('dataset/prediction/pet_2.png') #Make a prediction

     
    
    
    
       

