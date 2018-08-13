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

import urllib2
import re
import json

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

headers = {'X-Access-Token':'pPuub*lxup*cBsGU15fgDKuMRA75uIVn'}
xtilt_list = []
ytilt_list = []
water_percent2 = []
water_percent3 = []

'''
get WML environment and define the out_path
'''
result_file = 'result.csv'
last_time = 'last_time.json'
if environ.get('RESULT_DIR') is not None:
    output_result_folder = os.path.join(os.environ["RESULT_DIR"], "predict_result")
    output_result_path = os.path.join(output_result_folder, result_file)
    output_json_folder = os.path.join(os.environ["RESULT_DIR"], "last_time")
    output_json_path = os.path.join(output_result_folder, lsat_time)
else:
    output_result_folder = "predict_result"
    output_result_path = os.path.join("predict_result", result_file)
    output_json_folder = "last_time"
    output_json_path = os.path.join("last_time", last_time)
os.makedirs(output_result_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)

'''
get data
'''
req = urllib2.Request('http://openapi.ecois.info/v2/poi/device/data?sn=18031400075227&nodes=1,2,3&params=82,182,185&begin=20180615&end=20180801',headers=headers)
res_data = urllib2.urlopen(req)
res = res_data.read()
st = json.loads(res)
data1 = pd.DataFrame(st)
with open('.wktest-master/last_time.json') as load_f:
    json_dict = json.load(load_f)
print(json_dict)
json_key = json_dict.values()[0]
print(json_key)
key_num = 0
for i in range(len(data1.index)):
    if data1.index[i] == json_key:
        key_num = i
        break
print(key_num)
data_50 = data1.iloc[key_num:key_num+50,:]
for i in range(data_50.shape[0]):
    if len(data_50.iloc[i,0]['1'])==2:
        xtilt_list.append(float(data_50.iloc[i,0]['1']['182']))
        ytilt_list.append(float(data_50.iloc[i,0]['1']['185']))
print(len(xtilt_list),len(ytilt_list))
for i in range(data_50.shape[0]):
    water_percent2.append(float(data_50.iloc[i,0]['2']['82']))
    water_percent3.append(float(data_50.iloc[i,0]['3']['82']))
print(len(water_percent2),len(water_percent3))
data = pd.DataFrame({'xtilt':xtilt_list,'ytilt':ytilt_list,'water2':water_percent2,'water3':water_percent3})


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
#data = data.iloc[-50:,1:]
data = np.array(data)
x = []
x.append(data)

result = model.predict(np.array(x)) #predict with the datasets depend on the model,and the shape of result will be equal to (1,output_size)
print(result)
print(result.shape)

result_lst = result[0]
result_df = DataFrame({'predict':result_lst})
result_df.to_csv(output_result_path)

last_time = data_50.index[-1]
json_dict['last_time'] = last_time
print(json_dict)
with open(output_json_path,'w') as json_f:
    json.dump(json_dict,json_f)