from keras.datasets import cifar10

# Fetch the data:
(X, y), (_, _) = cifar10.load_data()

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# For matrix operations and dataframes.
import numpy as np

# Data visualizaton.
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import random as rn
 
# Configure some defaults.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

# Useful deep learning functions.
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Powerful deep learning module.
import tensorflow as tf

# For dealing with data.
import numpy as np  

fig, ax = plt.subplots(2, 5)
fig.set_size_inches(10, 6)

for i in range(2):
    for j in range(5):
        c = j + 5*i # Class counter
        l = np.random.choice(np.where(y == c)[0], 1)[0] # Get a random image from class c
        ax[i, j].imshow(X[l])
        ax[i, j].set_title('Class: ' + str(y[l]))
        # Hide grid lines
        ax[i, j].grid(False)
        # Hide axes ticks
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        
plt.tight_layout()

print('X (images)', X.shape)
print('y (classes)', y.shape)

# One-hot encode those integer values of class labels
y = to_categorical(y, 10)

# Normalize all entries to the interval [0, 1]
X = X / 255.

# defined architecture of the deep neural network.
model = Sequential()

model.add(Conv2D(filters = 16,      
                 kernel_size = (3, 3), 
                 padding = 'None',
                 activation = 'relu', 
                 input_shape = (32, 32, 3)))
model.add(Conv2D(filters = 32,      
                 kernel_size = (3, 3), 
                 padding = 'None',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 128,      
                 kernel_size = (3, 3), 
                 padding = 'Same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 128,      
                 kernel_size = (3, 3), 
                 padding = 'Same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 256,      
                 kernel_size = (3, 3), 
                 padding = 'Same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.7))

# A dense layer with 10 neurons (one per class).x 
model.add(Dense(10, activation = "softmax"))
model.summary()

batchsize = 300

epochs = 50

opt = Adam(lr = 0.001)

model.compile(optimizer = opt,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X, 
                    y,
                    batch_size = batchsize,
                    epochs = epochs, 
                    validation_split = 0.2, # DON'T CHANGE validation_split!
                    verbose = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()