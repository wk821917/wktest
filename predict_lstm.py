import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pandas import Series, DataFrame

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

data = pd.read_csv('wktest-master/predict_data.csv')
f = open('wktest-master/lstm.json', 'r')
json_string = f.read()
f.close()

model = model_from_json(json_string)
model.load_weights('wktest-master/weight.hdf5')
# print(json_string)

input_step_size = 50
output_size = 10
inputs = []
outputs = []
data = data.iloc[:,1:]

test_data = data.iloc[-50:,:]
result = model.predict(test_data)
print(result)
print(result.shape)
