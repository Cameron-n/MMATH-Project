from tensorflow.keras.callbacks import Callback 
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

class confusion__matrix(Callback):

    def __init__(self, X_val, Y_val, name, data, end, dif=1, weights_list=[], fine_tune=0):
        self.X_val = X_val
        self.Y_val = Y_val
        self.data = data
        self.name = name
        self.end = end
        self.dif = dif
        self.weights_list = weights_list
        self.fine_tune = fine_tune

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}): 
        if (epoch+1)%self.dif == 0:
            
            if self.fine_tune == False:
                current_weights = self.model.layers[-1].get_weights()
            else:
                current_weights = self.model.get_weights()
            
            self.weights_list += current_weights
            
            pred = self.model.predict(self.X_val)
            max_pred = np.argmax(pred, axis=1)
            max_y = np.argmax(self.Y_val, axis=1)
            cnf_mat = confusion_matrix(max_y, max_pred)
            self.data = np.concatenate([self.data,cnf_mat])
            if epoch+1 == self.end:
                np.savetxt(f'{self.name}',self.data)
                pickle.dump(self.weights_list, open(f'{self.name}_weights', 'wb'))
            
#Try to train in (or subset of it) using this very machine device!
#Try random network
#Overtrain pretrained network on tiny subset
#Give up
#Test if .fit in a loop actually really truely very much so works