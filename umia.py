'''
Created on 05 May 2021

@author: Yuefeng Peng
Based on https://github.com/AhmedSalem2/ML-Leaks/blob/master/mlLeaks.py
'''

import sys

sys.dont_write_bytecode = True

import numpy as np

import pickle
from sklearn.model_selection import train_test_split
import random
import lasagne
import os
from sklearn.metrics import roc_auc_score
import argparse
import deeplearning as dp
import classifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import  tensorflow as tf
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (CIFAR10 or News)')
parser.add_argument('--classifierType', default='cnn', help='Which classifier cnn or nn')
parser.add_argument('--dataFolderPath', default='./data/', help='Path to store data')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs to train shadow/target models')
parser.add_argument('--preprocessData', action='store_true',
                    help='Preprocess the data, if false then load preprocessed data')
parser.add_argument('--trainTargetModel', action='store_true',
                    help='Train a target model, if false then load an already trained model')
parser.add_argument('--clip_k', default=2,
                    help='Parameter k in clipping')
parser.add_argument('--rescale_t', default=100,
                    help='Parameter T in rescaling')


opt = parser.parse_args()

def softmax_t(x, t):
	x_row_max = x.max(axis=-1)
	x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
	x = x - x_row_max
	x_exp = np.exp(x / t)
	x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
	softmax = x_exp / x_exp_row_sum
	return softmax

# Picking the top X probabilities
def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)


def readCIFAR10():
	(X, y), (XTest, yTest) = tf.keras.datasets.cifar10.load_data()
	y=np.squeeze(y)
	yTest=np.squeeze(yTest)
	X = np.vstack((X, XTest))
	y = np.concatenate((y, yTest))
	print(X.shape)
	print(y.shape)
	return X, y, XTest, yTest

def readCIFAR100():
	(X, y), (XTest, yTest) = tf.keras.datasets.cifar100.load_data()
	y=np.squeeze(y)
	yTest=np.squeeze(yTest)
	X = np.vstack((X, XTest))
	y = np.concatenate((y, yTest))
	print(X.shape)
	print(y.shape)
	return X, y, XTest, yTest

def readMNIST():
	(X, y), (XTest, yTest) = tf.keras.datasets.mnist.load_data()
	y=np.squeeze(y)
	yTest=np.squeeze(yTest)
	print(X.shape)
	print(y.shape)
	X = X.reshape(-1, 1, 28, 28)
	XTest = XTest.reshape(-1, 1, 28, 28)
	print(X.shape)
	print(y.shape)
	return X, y, XTest, yTest

def readPurchase():
	X, y = [], []
	with open('data/SourceDatasets/dataset_purchase', 'r') as infile:
		reader = csv.reader(infile)
		for line in reader:
			y.append(int(line[0]))
			X.append([int(x) for x in line[1:]])
		X = np.array(X)
		y = np.array(y) - 1
	print(X.shape)
	print(y.shape)
	return X,y

def readTexas():
	X, y = [], []
	with open('data/SourceDatasets/texas/100/feats', 'r') as infile:
		reader = csv.reader(infile)
		for line in reader:
			X.append([int(x) for x in line[1:]])
		X = np.array(X)
	with open('data/SourceDatasets/texas/100/labels', 'r') as infile:
		reader = csv.reader(infile)
		for line in reader:
			y.append(int(line[0]))
		y = np.array(y) - 1
	print(X.shape)
	print(y.shape)
	return X,y

def readLocation():
	X, y = [], []
	with open('data/SourceDatasets/location', 'r') as infile:
		reader = csv.reader(infile)
		for line in reader:
			y.append(int(line[0]))
			X.append([int(x) for x in line[1:]])
		X = np.array(X)
		y = np.array(y) - 1
	print(X.shape)
	print(y.shape)
	return X,y

def readAdult():
	X_train, y_train  = load_data( 'data/SourceDatasets/adult/AdultTrain.npz')
	X_test,  y_test   = load_data( 'data/SourceDatasets/adult/AdultTest.npz')
	X = np.concatenate([X_train, X_test], axis=0)
	y = np.concatenate([y_train, y_test], axis=0)
	y = np.argmax(y, axis=1)
	print(X.shape)
	print(y.shape)
	return X,y


def trainTarget(modelType, X, y,
                X_test=[], y_test=[],
                splitData=True,
                test_size=0.5,
                inepochs=50, batch_size=300,
                learning_rate=0.001):
    if (splitData):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train = X
        y_train = y

    dataset = (X_train.astype(np.float32),
               y_train.astype(np.int32),
               X_test.astype(np.float32),
               y_test.astype(np.int32))

    attack_x, attack_y, theModel = dp.train_target_model(dataset=dataset, epochs=inepochs, batch_size=batch_size,
                                                         learning_rate=learning_rate,
                                                         n_hidden=128, l2_ratio=1e-07, model=modelType)

    return attack_x, attack_y, theModel


def load_data(data_name):
    with np.load(data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y


def trainAttackModel(X_test, y_test, clip_k, rescale_t):


    X_test = clipDataTopX(X_test, clip_k)
    X_test=X_test.astype(np.float32)
    y_test=y_test.astype(np.int32)

    X_train_t = softmax_t(np.log(X_test), rescale_t)
    # X_train_t=X_train
    model = KMeans(n_clusters=2)
    model.fit(X_train_t)

    y_pre = model.predict(X_train_t)

    if np.amax(model.cluster_centers_[0]) > np.amax(model.cluster_centers_[1]):
        y_pre = 1 - y_pre
    print('Testing Accuracy: {}'.format(accuracy_score(y_test, y_pre)))
    print('More detailed results:')
    print(classification_report(y_test, y_pre))
    print(model.cluster_centers_)




def preprocessingMNIST(toTrainData, toTestData):
	def reshape_for_save(raw_data):
		return raw_data.astype(np.float32)

	offset = np.mean(reshape_for_save(toTrainData), 0)
	scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

	def rescale(raw_data):
		return (reshape_for_save(raw_data) - offset) / scale

	return rescale(toTrainData), rescale(toTestData)

def preprocessingCIFAR(toTrainData, toTestData):
	def reshape_for_save(raw_data):
		raw_data = raw_data.transpose(0, 3, 1, 2)
		return raw_data.astype(np.float32)

	offset = np.mean(reshape_for_save(toTrainData), 0)
	scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

	def rescale(raw_data):
		return (reshape_for_save(raw_data) - offset) / scale

	return rescale(toTrainData), rescale(toTestData)

def shuffleAndSplitData(dataX, dataY,cluster):
	c = zip(dataX, dataY)
	c = list(c)
	random.shuffle(c)
	dataX, dataY = zip(*c)
	toTrainData  = np.array(dataX[:cluster])
	toTrainLabel = np.array(dataY[:cluster])

	shadowData  = np.array(dataX[cluster:cluster*2])
	shadowLabel = np.array(dataY[cluster:cluster*2])

	toTestData  = shadowData[:]
	toTestLabel = shadowLabel[:]

	shadowTestData  = toTrainData[:]
	shadowTestLabel = toTrainLabel[:]

	return toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel


def initializeData(dataset,dataFolderPath = './data/'):
	if (dataset == 'Adult'):
		print("Loading data")
		dataX, dataY = readAdult()
		print("Preprocessing data")
		cluster = 10000
		dataPath = dataFolderPath + dataset + '/Preprocessed'
		toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
			dataX, dataY, cluster)
		toTrainDataSave, toTestDataSave = toTrainData, toTestData
		shadowDataSave, shadowTestDataSave = shadowData, shadowTestData

	if (dataset == 'Texas'):
		print("Loading data")
		dataX, dataY = readTexas()
		print("Preprocessing data")
		cluster = 10000
		dataPath = dataFolderPath + dataset + '/Preprocessed'
		toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
			dataX, dataY, cluster)
		toTrainDataSave, toTestDataSave = toTrainData, toTestData
		shadowDataSave, shadowTestDataSave = shadowData, shadowTestData

	if (dataset == 'Location'):
		print("Loading data")
		dataX, dataY = readLocation()
		print("Preprocessing data")
		cluster = 1200
		dataPath = dataFolderPath + dataset + '/Preprocessed'
		toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = shuffleAndSplitData(
			dataX, dataY, cluster)
		toTrainDataSave, toTestDataSave = toTrainData, toTestData
		shadowDataSave, shadowTestDataSave = shadowData, shadowTestData

	if(dataset == 'Purchase'):
		print("Loading data")
		dataX, dataY = readPurchase()
		print("Preprocessing data")
		cluster = 10000
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = toTrainData, toTestData
		shadowDataSave, shadowTestDataSave = shadowData, shadowTestData

	if(dataset == 'MNIST'):
		print("Loading data")
		dataX, dataY, _, _ = readMNIST()
		print("Preprocessing data")
		cluster = 10000
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingMNIST(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingMNIST(shadowData, shadowTestData)

	if(dataset == 'CIFAR10'):
		print("Loading data")
		dataX, dataY, _, _ = readCIFAR10()
		print("Preprocessing data")
		cluster = 10000
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingCIFAR(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)

	if(dataset == 'CIFAR100'):
		print("Loading data")
		dataX, dataY, _, _ = readCIFAR100()
		print("Preprocessing data")
		cluster = 10000
		dataPath = dataFolderPath+dataset+'/Preprocessed'
		toTrainData, toTrainLabel,shadowData,shadowLabel,toTestData,toTestLabel,shadowTestData,shadowTestLabel = shuffleAndSplitData(dataX, dataY,cluster)
		toTrainDataSave, toTestDataSave    = preprocessingCIFAR(toTrainData, toTestData)
		shadowDataSave, shadowTestDataSave = preprocessingCIFAR(shadowData, shadowTestData)

	try:
		os.makedirs(dataPath)
	except OSError:
		pass
	print(toTrainDataSave.shape, toTestDataSave.shape, shadowDataSave.shape, shadowTestDataSave.shape)

	np.savez(dataPath + '/targetTrain.npz', toTrainDataSave, toTrainLabel)
	np.savez(dataPath + '/targetTest.npz',  toTestDataSave, toTestLabel)
	np.savez(dataPath + '/shadowTrain.npz', shadowDataSave, shadowLabel)
	np.savez(dataPath + '/shadowTest.npz',  shadowTestDataSave, shadowTestLabel)

	print("Preprocessing finished\n\n")


def initializeTargetModel(dataset, num_epoch, dataFolderPath='./data/', modelFolderPath='./model/',
                          classifierType='cnn'):
    dataPath = dataFolderPath + dataset + '/Preprocessed'
    attackerModelDataPath = dataFolderPath + dataset + '/attackerModelData'
    modelPath = modelFolderPath + dataset
    try:
        os.makedirs(attackerModelDataPath)
        os.makedirs(modelPath)
    except OSError:
        pass
    print("Training the Target model for {} epoch".format(num_epoch))
    targetTrain, targetTrainLabel = load_data(dataPath + '/targetTrain.npz')
    targetTest, targetTestLabel = load_data(dataPath + '/targetTest.npz')
    attackModelDataTarget, attackModelLabelsTarget, targetModelToStore = trainTarget(classifierType, targetTrain,
                                                                                     targetTrainLabel,
                                                                                     X_test=targetTest,
                                                                                     y_test=targetTestLabel,
                                                                                     splitData=False,
                                                                                     inepochs=num_epoch, batch_size=100)
    np.savez(attackerModelDataPath + '/targetModelData.npz', attackModelDataTarget, attackModelLabelsTarget)
    np.savez(modelPath + '/targetModel.npz', *lasagne.layers.get_all_param_values(targetModelToStore))
    return attackModelDataTarget, attackModelLabelsTarget


def generateAttackData(dataset, classifierType, dataFolderPath, num_epoch, preprocessData,
                       trainTargetModel):
    attackerModelDataPath = dataFolderPath + dataset + '/attackerModelData'
    if (preprocessData):
        initializeData(dataset)

    if (trainTargetModel):
        targetX, targetY = initializeTargetModel(dataset, num_epoch, classifierType=classifierType)
    else:
        targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')

    return targetX, targetY


def attack_UMIA(dataset='CIFAR10', classifierType='cnn', dataFolderPath='./data/', num_epoch=50, preprocessData=True,
                trainTargetModel=True, clip_k=2, rescale_t=100):
    targetX, targetY= generateAttackData(dataset, classifierType, dataFolderPath,
                                                            num_epoch, preprocessData, trainTargetModel)
    print("Training the attack model in an unsupervised way")
    trainAttackModel(targetX, targetY, clip_k, rescale_t)


attack_UMIA(dataset= opt.dataset,classifierType = opt.classifierType,dataFolderPath=opt.dataFolderPath,num_epoch = opt.num_epoch,preprocessData=opt.preprocessData,trainTargetModel = opt.trainTargetModel, clip_k=opt.clip_k, rescale_t=opt.rescale_t)

