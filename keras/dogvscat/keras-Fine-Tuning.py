from keras import Model
from sklearn.datasets import load_files
from sklearn.metrics import log_loss
from keras.utils import np_utils
import numpy as np
import os
from glob import glob
from keras.preprocessing import image
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import backend as K


def logloss(y_true, y_pred):

    # Prepare numpy array data
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # assert (len(y_true) and len(y_true) == len(y_pred))

    # Clip y_pred between eps and 1-eps
    # p = np.clip(y_pred, eps, 1-eps)
    # loss = np.mean(np.sum(y_true*np.log(p)))
    p = K.clip(y_pred, K.epsilon(), 1)
    loss = K.mean(-K.sum(y_true*K.log(p),axis=1))

    return loss

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(299, 299))
    # 将PIL.Image.Image类型转化为格式为(299, 299, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 299, 299, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 2)
    return dog_files, dog_targets

InceptionV3_model = InceptionV3(weights='imagenet',include_top=False)
x = GlobalAveragePooling2D()(InceptionV3_model.output)
InceptionV3_model_g = Model(InceptionV3_model.input,x)

def cache_dataset(path):
    cachefilepath = os.path.join('cache',path)
    data = load_files(path)
    files = np.array(data['filenames'])
    cache_files = []
    targets = np_utils.to_categorical(np.array(data['target']), 2)
    for file in tqdm(files):
        cachefilename = 'cache/' + file + '.npy'
        if not os.path.exists(cachefilepath):
        #if 1:
            filetensor = path_to_tensor(file).astype('float32') / 255
            cache_tensor = InceptionV3_model_g.predict(filetensor)
            np.save(cachefilename,cache_tensor)
        cache_files.append(cachefilename)
    return np.array(cache_files),targets

train_files, train_targets = cache_dataset('kera_data/train')
valid_files, valid_targets = cache_dataset('kera_data/valid')
#test_files, test_targets = cache_dataset('kera_data/test')

class_name = [item[16:-1] for item in sorted(glob("kera_data/train/*/"))]

def load_cache_date(path):
    list_of_tensors = []
    for cache_file in path:
        list_of_tensors.append(np.load(cache_file))
    return np.vstack(list_of_tensors)


train_tensors = load_cache_date(train_files)
valid_tensors = load_cache_date(valid_files)
#test_tensors = load_cache_date(test_files)

final_model = Sequential()
#final_model.add(GlobalAveragePooling2D(input_shape=InceptionV3_model.output))
final_model.add(Dense(1000,activation='relu',input_shape=(2048,)))
final_model.add(BatchNormalization(axis=-1))
final_model.add(Dense(500,activation='relu'))
final_model.add(BatchNormalization(axis=-1))
# final_model.add(Dense(250,activation='relu'))
# final_model.add(BatchNormalization(axis=-1))
# final_model.add(Dense(120,activation='relu'))
# final_model.add(BatchNormalization(axis=-1))
# final_model.add(Dense(60,activation='relu'))
# final_model.add(BatchNormalization(axis=-1))
# final_model.add(Dense(30,activation='relu'))
# final_model.add(BatchNormalization(axis=-1))
# final_model.add(Dense(10,activation='relu'))
# final_model.add(BatchNormalization(axis=-1))
final_model.add(Dense(2,activation='sigmoid'))
final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',logloss])

checkpointer = ModelCheckpoint(filepath='keras_models/weights.best.Inceptionv3.hdf5',
                               verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=5,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=100,     # 用多大量的数据计算直方图
                 write_graph=False,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

#final_model.load_weights('keras_models/weights.best.Inceptionv3.hdf5')
final_model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=50, batch_size=100, callbacks=[checkpointer,tb,es], verbose=1)
#
# final_model.load_weights('keras_models/weights.best.Inceptionv3.hdf5')
# predictions = [np.argmax(final_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
# # predictions = [final_model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in test_tensors]
# # test_loss = logloss(test_targets,np.array(predictions))
# print('Test accuracy: %.4f%%' % test_accuracy)