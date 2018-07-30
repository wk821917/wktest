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

result_file = 'result.csv'
if environ.get('RESULT_DIR') is not None:
    output_result_folder = os.path.join(os.environ["RESULT_DIR"], "predict_result")
    output_result_path = os.path.join(output_result_folder, result_file)
else:
    output_result_folder = "predict_result"
    output_result_path = os.path.join("predict_result", model_filename)
    
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
data = data.iloc[-50:,1:]
data = np.array(data)
x = []
x.append(data)
result = model.predict(np.array(x))
print(result)
print(result.shape)

result_lst = result[0]
result_df = DataFrame({'predict':result_lst})
result_df.to_csv(output_result_path)
