# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:45:45 2021

@author: Cameron
"""

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Currently there is no code for generating the 3D surface plots
# automatically so I just ran the network manually and wrote down the
# accuracy for each stage.
# This would be the longest section to write code for.

resnet50_time = {'avg_time' : [8,27.6,110], 
            'time' : [
            [0.2787,0.5526,0.6480,0.6987],
            [0.4599,0.6137,0.6907,0.7387,0.7748],
            [0.4545,0.5954,0.6557,0.7232,0.7674],
            [0.4555,0.5971,0.6364,0.6824,0.7447,0.7858]],
            'val_time' : [
            [0.4671,0.6013,0.6566,0.6923],
            [0.5550,0.6325,0.6744,0.7009,0.7166],
            [0.5537,0.5971,0.6591,0.6895,0.7133],
            [0.5580,0.5943,0.6164,0.6754,0.7018,0.7137]],
            }

densenet121_time = {'avg_time' : [5.9,22,100],
               'time' : [
                [0.2903,0.5589,0.6645,0.7152,0.7476],
                [0.3863,0.6049,0.6941,0.7407,0.7706],
                [0.3858,0.5711,0.6516,0.7183,0.7569,0.7806],
                [0.3874,0.5696,0.6188,0.6711,0.7300,0.7638,0.7875],
                [0.3833,0.5682,0.6157,0.6405,0.6858,0.7395,0.7700],
                [0.3852,0.5708,0.6169,0.6424,0.6572,0.6974,0.7462,0.7737]],
                'val_time' : [
                [0.4695,0.6218,0.6787,0.7124,0.7369],
                [0.5248,0.6571,0.7071,0.7353,0.7541],
                [0.5262,0.5885,0.6850,0.7200,0.7429,0.7613],
                [0.5263,0.5911,0.6195,0.6956,0.7263,0.7489,0.7629],
                [0.5249,0.5882,0.6181,0.6341,0.7008,0.7321,0.7488],
                [0.5264,0.5916,0.6219,0.6326,0.6470,0.7100,0.7337,0.7526]]
                }

densenet169_time = {'avg_time' : [7.4,27,97],
               'time' : [
                [0.3606,0.6594,0.7377,0.7739],
                [0.4277,0.6594,0.7433,0.7833],
                [0.4275,0.6090,0.6900,0.7531,0.7800],
                [0.4233,0.6060,0.6409,0.7027,0.7598,0.7925]],
                'val_time' : [
                [0.5851,0.7010,0.7419,0.7631],
                [0.5759,0.7067,0.7493,0.7754],
                [0.5766,0.6209,0.7223,0.7531,0.7717],
                [0.5753,0.6213,0.6411,0.7264,0.7569,0.7728]],
                }


mobilenetv2_time = {'avg_time' : [6,13.5,30], 
               'time' : [
                [0.2183,0.4390],
                [0.1604,0.3466,0.5129],
                [0.1616,0.2350,0.3891],
                [0.1604,0.2251,0.2590,0.3951]],
               'val_time' : [
                [0.3951,0.5170],
                [0.2101,0.4515,0.5488],
                [0.2115,0.2492,0.4739],
                [0.2080,0.2485,0.2631,0.4815]]
                }

mobilenetv2 = { 
               'time' : [
                [0.2249,0.4538,0.5664,0.6165,0.6482],
                [0.1660,0.3435,0.5108,0.5819,0.6188,0.6458],
                [0.1621,0.2331,0.3835,0.5205,0.5753,0.6151,0.6420],
                [0.1494,0.2264,0.2597,0.3989,0.5263,0.5785,0.6149,0.6387],
                [0.1641,0.2375,0.2591,0.2687,0.4097,0.5250,0.5751,0.6070,0.6315],
                [0.1676,0.2343,0.2573,0.2687,0.2740,0.4140,0.5245,0.5757,0.6082,0.6315]],
               'val_time' : [
                [0.3526,0.5207,0.5845,0.6196,0.6406],
                [0.2103,0.4409,0.5465,0.5900,0.6191,0.6417],
                [0.2123,0.2493,0.4666,0.5518,0.5938,0.6204,0.6372],
                [0.2076,0.2495,0.2605,0.4833,0.5553,0.5952,0.6165,0.6384],
                [0.2160,0.2491,0.2606,0.2653,0.4846,0.5512,0.5841,0.6087,0.6280],
                [0.2120,0.2501,0.2613,0.2673,0.2702,0.4816,0.5445,0.5825,0.6065,0.6267]],
                'name' : 'mobilenetv2'
                }

resnet50 = {
            'time' : [
            [0.2772,0.5561,0.6491,0.7012,0.7353],
            [0.4574,0.6118,0.6911,0.7409,0.7772,0.8092],
            [0.4545,0.5984,0.6558,0.7253,0.7689,0.8022,0.8332],
            [0.4533,0.5963,0.6356,0.6836,0.7454,0.7867,0.8214,0.8509],
            [0.4550,0.5970,0.6349,0.6627,0.7029,0.7613,0.7993,0.8350,0.8638],
            [0.4544,0.5943,0.6351,0.6619,0.6797,0.7162,0.7725,0.8112,0.8441,0.8737]],
            'val_time' : [
            [0.4641,0.6026,0.6598,0.6920,0.7131],
            [0.5563,0.6368,0.6792,0.7009,0.7199,0.7283],
            [0.5549,0.5924,0.6606,0.6926,0.7081,0.7241,0.7334],
            [0.5591,0.5924,0.6179,0.6706,0.6978,0.7140,0.7242,0.7287],
            [0.5599,0.5932,0.6188,0.6308,0.6833,0.6992,0.7156,0.7250,0.7295],
            [0.5601,0.5983,0.6165,0.6289,0.6372,0.6875,0.7025,0.7153,0.7244,0.7327]],
            'name' : 'resnet50'
            }

def surface(data,val=True, save=False):
    """Creates surface plot from data.

    data -- data for surface plot
    val -- Set False to not use validation values (i.e. use actual accuracy from data)
    save -- Set True to save png
    """
    x = 6*[0,1,2,3,4,5]
    x.sort()
    y = 6*[0,1,2,3,4,5]
    if val == True:
        z = data['val_time']
    else:
        z = data['time']
    z = [i[-6:] for i in z]
    z[0].insert(0,0.1000)
    z = [item for sublist in z for item in sublist]
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel("Nonfine")
    ax.set_ylabel("Fine")
    ax.set_zlabel("Accuracy")
    plt.title(data['name'])
    ax.plot_trisurf(x,y,z,cmap=cm.coolwarm)
    
    if save == True:
        plt.savefig(f"{filepath_model}_surface_{filepath_dataset}.png",dpi=300)
    
# ignore

def fine(x,a,total_time):
    return (total_time - a*x[0])/x[1]

# ignore

def plott(x):
    x = x['val_time']
    x = [x[i][-1] for i in range(len(x))]
    plt.plot(x)
    
    
def deer(x):
    """In-progress! Plots confusion matrix.

    x -- data
    
    """
    fig, ax = plt.subplots()
    ax.matshow(x)
##    for (i, j), z in np.ndenumerate(x):
#        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
#    #plt.tight_layout()
    plt.show()
   
# ignore

def locate(x,a):
    value = np.partition((x - 50000*np.eye(10,10)).flatten(),-a)[-a]
    return np.where(x == value)

#ignore

def adder(s):
    x = np.loadtxt(s)
    x = x[10:]
    x = x.reshape(int(len(x)/10),10,10)
    x = x[-1]
    listy = []
    for j in range(len(x)):
        total = 0
        for i in x[j]:
            total += i
        total -= x[j,j]    
        listy.append(total)
    return listy

#ignore

def horse(x,y):
    x = x[10:]
    y = y[10:]
    #draw top 3 incorrect matches over time
    #label graph with cat, dog, deer etc
    pass
        
