import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import re 
import os
from os import environ
# import hickle as hkl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
K.clear_session()
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.layers import LSTM, SimpleRNN,LeakyReLU
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras import regularizers


data = pd.read_csv('wktest-master/actdata.csv')
input_step_size = 500
output_size = 10

model_filename = "lstm.json"
picname = 'pred_act.png'
# writing the train model and getting input data
if environ.get('RESULT_DIR') is not None:
    output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
    output_model_path = os.path.join(output_model_folder, model_filename)
    output_pic_folder = os.path.join(os.environ["RESULT_DIR"], "picture")
    output_pic_path = os.path.join(output_model_folder, picname)
else:
    output_model_folder = "model"
    output_model_path = os.path.join("model", model_filename)
    output_pic_folder = "picture"
    output_pic_path = os.path.join("picture", picname)

os.makedirs(output_model_folder, exist_ok=True)

os.makedirs(output_pic_folder, exist_ok=True)

#writing metrics
if environ.get('JOB_STATE_DIR') is not None:
    tb_directory = os.path.join(os.environ["JOB_STATE_DIR"], "logs", "tb", "test")
else:
    tb_directory = os.path.join("logs", "tb", "test")

os.makedirs(tb_directory, exist_ok=True)
tensorboard = TensorBoard(log_dir=tb_directory)

def dataset_setup(data):
    data = np.array(data)
    inputs = []
    outputs = []
    for i in range(len(data)-input_step_size-output_size):
        inputs.append(data[i:i + input_step_size])
        outputs.append(data[i + input_step_size: i + input_step_size+ output_size])
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    size1 = int(0.8*inputs.shape[0])
    size2 = int(0.9*inputs.shape[0])
    x_train  = inputs[:size1,:]
    y_train = outputs[:size1,:,1]
    x_val = inputs[size1:size2,:]
    y_val = outputs[size1:size2,:,1]
    x_test = inputs[size2:,:]
    y_test = outputs[size2:,:,1]
#     hkl.dump(x_test, './x_test.hkl')
#     hkl.dump(y_test, './y_test.hkl')
    plt.figure(figsize=(100,10))
    plt.plot(DataFrame(data))
    plt.savefig('./out_th.png')
    plt.clf()
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_model(x_train):
    m_inputs = Input(shape=(x_train.shape[1],x_train.shape[2]))
    lstm1 = LSTM(units=128, return_sequences=True)(m_inputs)
    #drop1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(units=64,return_sequences=True)(lstm1)
    #drop2 = Dropout(0.2)(lstm2)
    lstm3 = LSTM(units=32,return_sequences=True)(lstm2)
    #drop3 = Dropout(0.2)(lstm3)
    fa = Flatten()(lstm3)
    out = Dense(10)(fa)#kernel_regularizer=regularizers.l2(0.01)
    model = Model(m_inputs,out)
    model.compile(loss='mae', optimizer='adam')
    return model

def train_and_test_model(model,x_train, y_train, x_val, y_val, x_test, y_test):
    learn_rate = lambda epoch: 0.0001 if epoch < 5 else 0.00001
    callbacks = [LearningRateScheduler(learn_rate)]
    callbacks.append(ModelCheckpoint(filepath='./weights.hdf5', monitor='val_loss', save_best_only=True))	
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), verbose=1, shuffle=False, callbacks=callbacks)
    json_string = model.to_json()
    with open(output_model_path, "w") as f:
        f.write(json_string)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig('./loss.png')
    y_pred = model.predict(x_test)
    x = [i for i in range(y_pred.shape[0])]
    ax1 = plt.subplot(211)
    ax1.set_title('predict')
    ax1.plot(x,y_pred[:,0])
    ax2=plt.subplot(212)
    ax2.set_title('active')
    ax2.plot(x,y_test[:,0])
    plt.tight_layout(2)
    plt.savefig(output_pic_path)


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_setup(data.iloc[:,1:])
    model = create_model(x_train)
    train_and_test_model(model,x_train, y_train, x_val, y_val, x_test, y_test)
#     model.save(output_model_path)
