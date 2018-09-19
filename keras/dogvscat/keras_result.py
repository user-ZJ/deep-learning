import csv
import random

import tensorflow as tf
import os
import numpy as np
import tensorflow.contrib.slim as slim
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.datasets import load_files
from sklearn.metrics import log_loss
from keras.utils import np_utils
from tqdm import tqdm
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras import Model, Input

InceptionV3_model = InceptionV3(weights='imagenet',include_top=False)
x = GlobalAveragePooling2D()(InceptionV3_model.output)
InceptionV3_model_g = Model(InceptionV3_model.input,x)

def logloss(y_true, y_pred,eps=1e-15):

    # Prepare numpy array data
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # assert (len(y_true) and len(y_true) == len(y_pred))

    # Clip y_pred between eps and 1-eps
    p = np.clip(y_pred, eps, 1-eps)
    loss = np.mean(-np.sum(y_true*np.log(p),axis=1))

    return loss

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(299, 299))
    # 将PIL.Image.Image类型转化为格式为(299, 299, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 299, 299, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def cache_dataset(path):
    data = load_files(path)
    cachefilepath = os.path.join('cache', path)
    files = np.array(data['filenames'])
    cache_files = []
    targets = np_utils.to_categorical(np.array(data['target']), 2)
    for file in tqdm(files):
        cachefilename = 'cache/' + file + '.npy'
        if not os.path.exists(cachefilepath):
        #if 1:
            filetensor = path_to_tensor(file).astype('float32') / 255
            cache_tensor = InceptionV3_model_g.predict(filetensor)
            np.save(cachefilename, cache_tensor)
        cache_files.append(cachefilename)
    return np.array(cache_files),targets

def load_cache_date(path):
    list_of_tensors = []
    for cache_file in path:
        list_of_tensors.append(np.load(cache_file))
    return np.vstack(list_of_tensors)

test_files, test_targets = cache_dataset('kera_data/upload_data')

test_tensors = load_cache_date(test_files)

input = Input(shape=(2048,))
x = Dense(1000,activation='relu')(input)
x = BatchNormalization(axis=-1)(x)
x = Dense(500,activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
# x = Dense(250,activation='relu')(x)
# x = BatchNormalization(axis=-1)(x)
# x = Dense(120,activation='relu')(x)
# x = BatchNormalization(axis=-1)(x)
# x = Dense(60,activation='relu')(x)
# x = BatchNormalization(axis=-1)(x)
# x = Dense(30,activation='relu')(x)
# x = BatchNormalization(axis=-1)(x)
# x = Dense(10,activation='relu')(x)
# x = BatchNormalization(axis=-1)(x)
x = Dense(2,activation='sigmoid')(x)
final_model = Model(input=input, output=x)
final_model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
final_model.summary()
final_model.load_weights('keras_models/weights.best.Inceptionv3.hdf5')
predictions = []
prd = []
for tensor in test_tensors:
    prediction = final_model.predict(np.expand_dims(tensor, axis=0))
    prd.append(prediction[0])
    predictions.append(prediction[0][1])

print(log_loss(test_targets,np.array(prd)))
with open('kera_data/upload_data/result.csv','w',newline='') as f:
    writer = csv.writer(f)
    header = ['id','label']
    writer.writerow(header)
    for i in range(len(test_files)):
        row = [os.path.basename(test_files[i]).split('.')[0],predictions[i]]
        writer.writerow(row)
