import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import re
import os
from os import environ
from matplotlib import pyplot as plt
import scipy
from sklearn.cluster import KMeans

filename = 'kmeans_wtr_tilt.csv'
picname = 'level_pic.png'
if environ.get('RESULT_DIR') is not None:
    output_file_folder = os.path.join(os.environ["RESULT_DIR"], "result_w_t")
    output_file_path = os.path.join(output_file_folder, filename)
    output_pic_folder = os.path.join(os.environ["RESULT_DIR"], "picture")
    output_pic_path = os.path.join(output_model_folder, picname)
else:
    output_model_folder = "result_w_t"
    output_model_path = os.path.join("result_w_t", filename)
    output_pic_folder = "picture"
    output_pic_path = os.path.join("picture", picname)

os.makedirs(output_file_folder, exist_ok=True)
os.makedirs(output_pic_folder, exist_ok=True)

data = pd.read_csv('wktest-master/water-tilt.csv')
data = data.iloc[:5000,-2:]
print('the shape of data is %s'%str(data.shape))
data_calcu = data.iloc[:,0]**2 + data.iloc[:,1]**2

kmeans = KMeans(n_clusters=5)
kmeans.fit(DataFrame(data_calcu))
data['kind'] = kmeans.predic(DataFrame(data_calcu))
data['calcu'] = data_calcu

color_dict = {0:'orange',1:'gray',2:'blue',3:'green',4:'purple'}
color_list = []
for i in data.iloc[:,-2]:
    color_list.append(color_dict[i])
plt.scatter(data.index,data.iloc[:,-1],alpha=0.3,c=np.array(color_list))
plt.savefig(output_pic_path)

kdmin_dic = {}
for i in range(5):
    kdmin_dic[float(data[data['kind']==i].iloc[:,-1].max())] = i
data['level'] = np.nan

for i in range(5,0,-1):    
    for j in (data[data['kind']==kdmin_dict[min(kdmin_dict.keys())]].index):
        data.iloc[j,-1] = int(i)
    kdmin_dict.pop(min(kdmin_dict.keys()))

print(np.unique(data.iloc[:,-1]))

data.to_csv(output_file_folder)
