# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:48:40 2020

@author: Cameron
"""

# CHECK .KERAS JSON FILE SO CHANNELS IS LAST
#---------------------------------------------------------#
# Loads models pretrained on ImageNet and then saves them 
# to disk.
# There are six models currently available:
# 0 - ResNet50
# 1 - VGG16 - currently not working for cifar10->Imagenet
# 2 - VGG19 - currently not working for cifar10->Imagenet
# 3 - DenseNet121
# 4 - DenseNet169
# 5 - MobileNetV2
#---------------------------------------------------------#


from tensorflow.keras.applications import ResNet50,VGG16,VGG19,DenseNet121,DenseNet169,MobileNetV2 #2.3.1
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras import Model



### Section for selecting model ###
###-----------------------------------------------------------###

# List of available models. Easily adjustable

available_model = [
        [ResNet50, 'ResNet50'],
        [VGG16, 'VGG16'],
        [VGG19, 'VGG19'],
        [DenseNet121, 'DenseNet121'],
        [DenseNet169, 'DenseNet169'],
        [MobileNetV2, 'mobilenet_v2']
        ]


# User input to select model and file name to save to

which_model = available_model[int(input('which model? '))]
my_pretrained_model = which_model[0]                        # Model
name = which_model[1]                                       # File name

###-----------------------------------------------------------###



### Section for loading and modifying model ###
###-----------------------------------------------------------###

# Batchnorm fix to freeze them (i.e. stop updating weights)
# May not be needed now that this script does not train the
# network

class FrozenBatchNormalization(layers.BatchNormalization):
    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=False)

BatchNormalization = layers.BatchNormalization
layers.BatchNormalization = FrozenBatchNormalization

# Loads pretrained model without classification layer (the last one)
# Try except clause accounts for densenet models not accepting the
# layers argument and is not strictly necessary now that this script 
# does not involve training the network.

try:
    model_pretrained_base = my_pretrained_model(include_top = False, weights = None, input_shape=(32,32,3), layers=layers)
except Exception:
    model_pretrained_base = my_pretrained_model(include_top = False, weights = None, input_shape=(32,32,3))
    
# Restore batchnorm layer definition. Probably not needed
    
layers.BatchNormalization = BatchNormalization 

# Stops the pretrained model from adjusting its weights.
# Not needed, keras does not save this!

for layer in model_pretrained_base.layers:
    layer.trainable = False

# Adds on a new classification layer with 10 classes (cifar10's categories)
    
x = model_pretrained_base.output
x = GlobalMaxPooling2D()(x)
x = Dense(10, activation='softmax')(x)
model_pretrained = Model(model_pretrained_base.input,x)

###-----------------------------------------------------------###



### Section for saving model to disk ###
###-----------------------------------------------------------###

# Finalises the model with its optimizer, loss, and metrics. 
# Now ready to be trained

model_pretrained.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Sets the directory to look for models
### IMPORTANT !!! THE FOLLOWING LINE WILL NEED TO BE CHANGED FOR USE BY OTHERS. ###
filepath_front = r"C:\Users\Cameron\MMath Project\Models\ImageNet_to_MNIST\backups"
### IMPORTANT !!! ------------------------------------------------------------- ###

# Saves the model to disk

model_pretrained.save(filepath_front + f'\\model_pretrained_ImageNet_to_cifar10_{name}_None')
###-----------------------------------------------------------###