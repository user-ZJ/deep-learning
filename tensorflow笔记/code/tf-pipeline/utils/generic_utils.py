#coding=utf-8
import logging

import os


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