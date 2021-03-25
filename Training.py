# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:27:14 2020

@author: Cameron
"""

# CHECK .KERAS JSON FILE SO CHANNELS IS LAST
#---------------------------------------------------------#
# This tells keras that the data is in the format, 
# [height, width, channels]. This is model dependant and other 
# programs, such as pytorch, default to *first*, not last, 
# when making models. This will need to be accounted for
# if importing models using onnx.
#---------------------------------------------------------#

#---------------------------------------------------------#
# Trains models and then saves them to disk.
# There are six models currently available:
# 0 - ResNet50
# 1 - VGG16 - currently not working for cifar10->Imagenet
# 2 - VGG19 - currently not working for cifar10->Imagenet
# 3 - DenseNet121
# 4 - DenseNet169
# 5 - MobileNetV2
#---------------------------------------------------------#


from tensorflow.keras.applications import * #2.3.1
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
import pickle
from tensorflow.math import confusion_matrix
from confusion_matrix_callback import confusion__matrix
import numpy as np


# Selects which directory to look for models 

while True:
    filepath_dataset = input("Which dataset? ")
    if filepath_dataset == "c":
        filepath_dataset = "ImageNet_to_cifar10"
        break
    elif filepath_dataset == 'i':                   # Not needed now
        filepath_dataset = "cifar10_to_ImageNet"  
        break
    elif filepath_dataset == 'm':
        filepath_dataset = "ImageNet_to_MNIST"
        break
    else:
        print("?")
        print("")

# List of available models.
# Each model needs data to be preprocessed differently
        
available_model = [
        [resnet50.preprocess_input, 'ResNet50'],
        [vgg16.preprocess_input, 'VGG16'],
        [vgg19.preprocess_input, 'VGG19'],
        [densenet.preprocess_input, 'DenseNet121'],
        [densenet.preprocess_input, 'DenseNet169'],
        [mobilenet_v2.preprocess_input, 'mobilenet_v2']
        ]

# User input to select which model to use.
# Will currently only accept the six models above

while True:
    filepath_model = int(input("Which model? "))
    if filepath_model in range(6):
        preprocess = available_model[filepath_model][0]
        filepath_model = available_model[filepath_model][1]
        break
    else:
        print("?")
        print("")
        
# User input to select to train from:
# 0 - scratch. i.e. from the backup directory
# 1 - the last point the model was trained
# 2 - random weights
# CURRENTLY ONLY TESTED FOR '0'

while True:
    backup = int(input("Start from scratch? "))
    if backup == 2 or True or False:
        break
    else:
        print("?")
        print("")

# Sets the directory to look for models
### IMPORTANT !!! THE FOLLOWING LINE WILL NEED TO BE CHANGED FOR USE BY OTHERS. ###
filepath_front = r"C:\Users\Cameron\MMath Project\Models"
### IMPORTANT !!! ------------------------------------------------------------- ###

# Unique file number generator so that each use of this script
# saves a different set of data. It is simply a number in a file that
# counts up by 1 each time this script is used. Changing this file will
# cause problems that could potentially overwrite data.
# The creation and location of this file is specified in the README

num = np.loadtxt(open(filepath_front + r'\num.txt','rb'), delimiter=",")
np.savetxt(filepath_front + r'\num.txt', np.array(num+1))
num = str(int(num[0]))



### Section for selecting the model's file location based on user input. ###
###-----------------------------------------------------------###

# - filepath_front is the main directory.
# - filepath_middle is the subdirectory specifying the two datasets that the
# transfer learning is done on. e.g. ImageNet to cifar10.
# - filepath_back is the model itself

if backup == True:
    filepath_back = f"model_pretrained_{filepath_dataset}_{filepath_model}"
    filepath_middle = f"\\{filepath_dataset}\\backups\\"
    filepath = filepath_front + filepath_middle + filepath_back

elif backup == 2:
    filepath_back = f"model_pretrained_{filepath_dataset}_{filepath_model}_None"
    filepath_middle = f"\\{filepath_dataset}\\backups\\"
    filepath = filepath_front + filepath_middle + filepath_back
    
elif backup == False:
    
    # In progress. Attempt to select a model that has already been trained to
    # continue training
    
    while True:
        answer = input("Choose which previous model? ")
        if answer == "":
            prev = num - 1
            break
        elif int(answer) in range(int(num)):
            prev = int(answer)
            break
        else:
            print("?")
            print("")
            
    filepath_back = f"model_pretrained_{filepath_dataset}_{filepath_model}_trained_{prev}"
    filepath_middle = f"\\{filepath_dataset}\{filepath_model.lower()}\\"
    filepath = filepath_front + filepath_middle + filepath_back
    
else:
    print("Impossible!") # Just an unneeded sanity check
###-----------------------------------------------------------###



### Section for selecting the dataset based on user input ###
###-----------------------------------------------------------###
    
if filepath_dataset == "ImageNet_to_cifar10":

    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    x_train = preprocess(x_train) #Normalize to expected range
    x_test = preprocess(x_test)
    y_train = to_categorical(y_train) #One HOT encoding
    y_test = to_categorical(y_test)
    data = np.zeros([10,10])
    
elif filepath_dataset == "ImageNet_to_MNIST":

    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = preprocess(x_train) #Normalize to expected range
    x_test = preprocess(x_test)
    y_train = to_categorical(y_train) #One HOT encoding
    y_test = to_categorical(y_test)
    
    # Many models only accept a minimum of 32x32 RGB images.
    # This reshapes MNIST to 32x32 images by filling the borders with
    # zeros, and makes them 'RGB' by duplicating the single 
    # greyscale channel across the Red, Green, and Blue channels.
    
    x_train = x_train.reshape(60000,28,28,1)
    x_train = np.tile(x_train,3)
    x = np.zeros((x_train.shape[0],32,32,3))
    x[:,3:31,3:31]=x_train[:,:,:]
    x_train = x
    
    x_test = x_test.reshape(10000,28,28,1)
    x_test = np.tile(x_test,3)
    x = np.zeros((x_test.shape[0],32,32,3))
    x[:,3:31,3:31]=x_test[:,:,:]
    x_test = x
    
    data = np.zeros([10,10])
    
elif filepath_dataset == "cifar10_to_ImageNet": # Not needed now

    i=0
    batch_train = np.load(f'C:/Users/Cameron/Downloads/Imagenet32_train_npz/Imagenet32_train_npz/train_data_batch_{i+1}.npz')
    x_train = batch_train['data'][:50000]
    x_train.resize(x_train.shape[0],3,32,32)
    x_train = x_train - np.min(x_train) # Make non-negative
    x_train = x_train/np.max(x_train) # reduce to range [0,1]
    x_train[:,0,:,:] -= 0.4914 #Red
    x_train[:,1,:,:] -= 0.4822 #Green
    x_train[:,2,:,:] -= 0.4465 #Blue
    x_train[:,0,:,:] /= 0.2023 #Red
    x_train[:,1,:,:] /= 0.1994 #Green
    x_train[:,2,:,:] /= 0.2010 #Blue
    y_train = batch_train['labels'][:50000] - 1
    y_train = to_categorical(y_train)
  
    batch_test = np.load('C:/Users/Cameron/Downloads/Imagenet32_val_npz/Imagenet32_val_npz/val_data.npz')
    x_test = batch_test['data'][:10000]
    x_test.resize(x_test.shape[0],3,32,32)
    x_test = x_test - np.min(x_test)
    x_test = x_test/np.max(x_test)
    x_test[:,0,:,:] -= 0.4914
    x_test[:,1,:,:] -= 0.4822
    x_test[:,2,:,:] -= 0.4465
    x_test[:,0,:,:] /= 0.2023
    x_test[:,1,:,:] /= 0.1994
    x_test[:,2,:,:] /= 0.2010
    y_test = batch_test['labels'][:10000] - 1
    y_test = to_categorical(y_test)
    data = np.zeros([1000,1000])
    
else:
    print("Impossible!")
###-----------------------------------------------------------###
    
    
    
# Load model
    
model_pretrained = load_model(filepath)

# Freeze layers (keras does not save this property in the models).

for layer in model_pretrained.layers:
    layer.trainable = False

# Unfreeze the classification layer
    
model_pretrained.layers[-2].trainable = True #Probably unneeded
model_pretrained.layers[-1].trainable = True

# Recompile after freezing layers

model_pretrained.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Select how many fine and nonfine tuning epochs.
# Currently accepts a max of 100 epochs of each.
# Gives the choice of zero of either (but not both!).

while True:
    end = input("epochs? ")
    if int(end) in range(0,101):
        end = int(end)
        break
    else:
        print("?")
        print("")
        
while True:
    fine_tune = input("fine tune? ")
    if int(fine_tune) in range(2):
        fine_tune = int(fine_tune)
        if fine_tune == True:
            while True:
                end_f = input("fine epochs? ")
                if int(end_f) in range(1,101):
                    end_f = int(end_f)
                    print("")
                    break
                else:
                    print("?")
                    print("") 
        break
    
    else:
        print("?")
        print("")
        
# Directory to save data and trained model to
        
filepath = filepath_front + f"\\{filepath_dataset}\{filepath_model.lower()}\\"


### Section for training and saving the model and data ###
# Alot of code is potentially unneeded or inflexible
###-----------------------------------------------------------###

# For nonfine tuning, sets up the confusion matrix script and trains the model

if end != 0:
    confusionmatrix = confusion__matrix(X_val=x_train, Y_val=y_train, name=filepath + f'model_pretrained_{filepath_dataset}_{filepath_model}_cmatrix_{num}',data=data,end=end)

    model_pretrained_history = model_pretrained.fit(x_train, y_train, validation_data=(x_test,y_test) ,epochs=end, batch_size=512*2)#,callbacks=[confusionmatrix])

# For fine tuning, sets up the confusion matrix script and trains the model,
# unfreezing the entire network
    
if fine_tune == True:
    
    for layer in model_pretrained.layers:
        layer.trainable = True    
        
        # Since DenseNet does not accept the batchnorm fix, this freezes the batchnorm layers
        # seperately
        
        if filepath_model[0:8] == "DenseNet" and type(layer) == type(model_pretrained.layers[-4]):
            layer.trainable = False
    
    confusionmatrix = confusion__matrix(X_val=x_train, Y_val=y_train, name=filepath + f'model_pretrained_{filepath_dataset}_{filepath_model}_cmatrix_{num}_f',data=data,end=end_f, fine_tune=1)
    
    # Recompiles model with a lower learning rate since we do not want to destroy the initial weights
    # Set at 0.00001 but can be easily modified to be, for example, model or epoch dependant
    
    model_pretrained.compile(optimizer=Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=['accuracy'])
    model_pretrained_history_f = model_pretrained.fit(x_train, y_train, validation_data=(x_test,y_test) ,epochs=end_f, batch_size=512*2)#,callbacks=[confusionmatrix])
    #pickle.dump(model_pretrained_history_f.history, open(filepath + f'model_pretrained_{filepath_dataset}_{filepath_model}_history_{num}_f','wb'))

#pickle.dump(model_pretrained_history.history, open(filepath + f'model_pretrained_{filepath_dataset}_{filepath_model}_history_{num}','wb'))
#model_pretrained.save(filepath + f'model_pretrained_{filepath_dataset}_{filepath_model}_trained_{num}')
###-----------------------------------------------------------###    