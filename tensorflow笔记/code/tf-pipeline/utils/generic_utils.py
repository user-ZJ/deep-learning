#coding=utf-8
import logging

import os
import threading

import numpy as np


def initialize_logger(output_dir):
    """
    初始化日志
    :param output_dir:日志存放路径
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(threadName)-10.10s] [%(levelname)-4.4s]  %(message)s"

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if output_dir is not None:
        handler = logging.FileHandler(os.path.join(output_dir, "log_info.txt"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(os.path.join(output_dir, "log_debug.txt"))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def mkdir(dir_path):
    """
    创建目录
    :param dir_path:
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def list_files(base_path, filter_func):
    """
    获取目录下所有满足filter_func的文件列表
    :param base_path: 目录
    :param filter_func: 文件过滤函数
    :return:
    """
    for folder, subs, files in os.walk(base_path):
        for filename in files:
            if filter_func(os.path.join(folder, filename)):
                yield (os.path.join(folder, filename))


def cosine_similarity(x,y,norm=True):
    """
    计算向量x和y的余弦相似度
    :param norm:是否做归一化
    :return:
    """
    cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    if norm:
        return 0.5+0.5*cos
    else:
        return cos

def cosine_distances(x,y,norm=False):
    """
    计算向量x和y的余弦距离，余弦距离= 1 - 余弦相似度
    :param norm:是否做归一化
    :return:
    """
    cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    if norm:
        return 1-(0.5+0.5*cos)
    else:
        return 1-cos


def run_once(f):
    """
    函数f只运行一次
    :param f: 函数
    :return:
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            wrapper.value = f(*args, **kwargs)
            return wrapper.value
        else:
            return wrapper.value

    wrapper.has_run = False
    wrapper.value = None
    return wrapper

class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))
    return g

def get_available_gpus():
    """
        返回可用GPU列表
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



if __name__ == '__main__':
    print(cosine_similarity([1, 2, 2, 1, 1, 1, 0], [1, 2, 2, 1, 1, 2, 1]))