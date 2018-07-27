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
f = open('lstm.json', 'r')
json_string = f.read()
f.close()

model = model_from_json(json_string)
model.load_weights('weight.hdf5')

print(json_string)
