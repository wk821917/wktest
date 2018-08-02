import os
from os import environ
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

'''
get WML environment and define the out_path
'''
result_file = 'result.csv'
if environ.get('RESULT_DIR') is not None:
    output_result_folder = os.path.join(os.environ["RESULT_DIR"], "predict_result")
    output_result_path = os.path.join(output_result_folder, result_file)
else:
    output_result_folder = "predict_result"
    output_result_path = os.path.join("predict_result", result_file)
os.makedirs(output_result_folder, exist_ok=True)

data = pd.read_csv('wktest-master/predict_data.csv') #get input data
f = open('wktest-master/lstm.json', 'r')  #load the json file 
json_string = f.read()
f.close()

model = model_from_json(json_string)  #define model by the json string
model.load_weights('wktest-master/weight.hdf5')  #load weights for the model
# print(json_string)

'''
input_step_size and output_size have to be the same when the model trained
'''
input_step_size = 50
output_size = 10
'''
transform the datasets
'''
inputs = []
outputs = []
data = data.iloc[-50:,1:]
data = np.array(data)
x = []
x.append(data)

result = model.predict(np.array(x)) #predict with the datasets depend on the model,and the shape of result will be equal to (1,output_size)
print(result)
print(result.shape)

result_lst = result[0]
result_df = DataFrame({'predict':result_lst})
result_df.to_csv(output_result_path)
