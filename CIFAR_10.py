
'''
Author: Yash Zanwar
This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import tensorflow as tf
import keras
import pandas as pd

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#function to unpickle the data

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#unpickling 5 data packs
data_pack1=unpickle("data_batch_1");
data_pack2=unpickle("data_batch_2");
data_pack3=unpickle("data_batch_3");
data_pack4=unpickle("data_batch_4");
data_pack5=unpickle("data_batch_5");

#concatenating 5 datapacks into 1
data=np.concatenate((data_pack1[b'data'],data_pack2[b'data'],data_pack3[b'data'],data_pack4[b'data'],data_pack5[b'data']),axis=0)/255
labels=np.concatenate((data_pack1[b'labels'],data_pack2[b'labels'],data_pack3[b'labels'],data_pack4[b'labels'],data_pack5[b'labels']),axis=0)

#builing keras sequntial model by adding layers
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compiling the model with opimiser and loss function
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the model with data and 3 epochs
model.fit(np.asarray(data), np.asarray(labels), epochs=3)

#unpickling test data
test_pack=unpickle("test_batch");
test_data=test_pack[b'data']/255
test_labels=test_pack[b'labels']

#evaluating the test data on trained model
loss,acc=model.evaluate(test_data,test_labels)
print(loss,acc)