#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


from Speech_Model_Attention import Speech_Model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# #进行配置，使用70%的GPU
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# #config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# set_session(tf.Session(config=config))


datapath = ''
modelpath = os.path.join(os.getcwd(), 'speech_model_file')


if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

# system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
# if(system_type == 'Windows'):
# 	datapath = 'E:\\语音数据集'
# 	modelpath = modelpath + '\\'
# elif(system_type == 'Linux'):
# 	datapath = 'dataset'
# 	modelpath = modelpath + '/'
# else:
print('*[Message] Unknown System\n')
datapath = 'dataset'
modelpath = modelpath + '/'


batch_size=8
ms = Speech_Model(batch_size)

#ms.LoadModel(modelpath + 'speech_model24_e_0_step_327500.model')
ms.TrainModel(datapath, epoch = 100, batch_size = batch_size, save_step = 110000/16)

# from tqdm import tqdm,trange
# from time import sleep
#
# for i in tqdm(range(10), desc='1st loop'):
#     sleep(0.1)


