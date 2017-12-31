# -*- coding:utf-8 -*-  
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import struct
import jieba
import re
import chardet
import time

map_NER = {'LOC':1, 'PER':2, 'ORG':3, 'MISC':4}
max_sequence_length = 120

def get_key(dic, val):
	res =  [k for k,v in dic.items() if v == val]
	return '' if (len(res) == 0) else res[0]

def readChinsesWordEmbedding(filename):
	dic = {}
	tup = []
	i = 1; 
	with open(filename, 'r', encoding='utf-8',errors='ignore') as fp:
		lp = fp.readline().
		for ln in fp: 
			ln = ln.replace('\n','')
			#ln.decode('utf-8').encode('gbk')
			s = re.split(' ', ln)

			s[0] = s[0].replace('\n', '')
			dic[i] = s[0].replace(' ', '') 
			#print(dic[i])
			tmp_tup = []
			for j in range(1, len(s)):
				tmp_tup.append(float(s[j]))
			i += 1
			tup.append(tmp_tup) 

	return (dic, tup)
	

def readEnglishWordEmbedding(filename):
	dic = {}
	tup = []
	i = 1;
	with open(filename, 'r') as fp:
		for ln in fp: 
			ln = ln.replace('\n','')
			s = re.split(' ', ln)
			s[0] = s[0].replace('\n', '')
			dic[i] = s[0].replace(' ', '') 
			tmp_tup = []
			for j in range(1, len(s)):
				tmp_tup.append(float(s[j]))
			i += 1
			tup.append(tmp_tup)
	return (dic, tup)
	
def splitChineseSequence(filename, dic):
	#cutlist = ["，", ",", "！", "……", "!", "？", "?", "；", ";" ]
	#cutlist = r'\，\,\;\；.\。\?\？\!\！' 
	cutlist = r'\,|\，|\.|\。|\！|\!|\?|\？|\:|\：'
	fp = open(filename, 'r', encoding='utf-8', errors='ignore')
	print(filename)
	strAll = fp.read().replace('\n','')
	strAll = strAll.replace('\t', '')
	col = re.split(cutlist, strAll)
	res = []
	for st in col: 
		lins = re.split(' ', st) 
		reslist = [0 for i in range(max_sequence_length)]
		pro_lins = []
		for x in lins:
			if x != '':
				pro_lins.append(x) 
		i = -1
		for vob in pro_lins:
			# if(len(pro_lins) > 58):
			# 	print(len(pro_lins))
			i = i + 1
			idx = get_key(dic, vob)
			if idx == '':
				continue
			reslist[i] = idx
			
		res.append(reslist)

	fp.close()
	return res

def splitEnglishiSeqence(filename, dic):
	print(filename)
	cutlist = r'\.|\?|\,|\…'
	fp = open(filename, 'r')
	strAll = fp.read().replace('\n', '')
	strAll = strAll.replace('\t', '')
	col = re.split(cutlist, strALL)
	res = []
	for st in col:
		lins = re.split(' ', st)
		reslist = [0 for i in range(max_sequence_length)]
		i = -1
		pro_lins = []
		for x in lins:
			if x != '':
				pro_lins.append(x)
		for vob in pro_lins:
			i += 1
			idx = get_key(dic, vob)
			if idx == '':
				continue
			reslist[i] = idx
		res.append(reslist)
	fp.close()
	return res

def mapChineseNER(filename, dic, article):
	print (filename)
	res = [[0 for j in range(max_sequence_length)] for i in range(len(article))]
	map_file = {}
	with open(filename, 'r', encoding='utf-8', errors='ignore') as fp:
		for st in fp:
			st = st.replace('\n', '')
			st = re.split('\t', st)

			idx = get_key(dic, st[-1].strip(' ')) 
			if idx == '':
				idx = 0
			ydx = map_NER[st[1]]
			map_file[idx] = ydx 
		for i in range(len(article)):
			for j in range(max_sequence_length):
				if article[i][j] != 0:
					idx = article[i][j]
					# print ('idx', idx, dic[idx])
					if idx in map_file:
						ydx = map_file[idx]
					else:
						ydx = 0
					article[i][j] = ydx


	return res

def mapEnglishNER(filename, dic, article):
	print (filename)
	res = [[0 for j in range(max_sequence_length)] for i in range(len(article))]
	map_file = {}
	with open(filename, 'r') as fp:
		for st in fp:
			st = st.replace('\n', '')
			st = re.split('\t', st)
			idx = get_key(dic, st[-1].strip(' '))
			if idx == '':
				idx = 0
			ydx = map_NER[st[1]]
			map_file[idx] = ydx
		for i in range(len(article)):
			for j in range(max_sequence_length):
				if article[i][j] != 0:
					idx = article[i][j]
					if idx in map_file:
						ydx = map_file[idx]
					else:
						ydx = 0
					article[i][j] = ydx

	return res

def loadChineseData():
	path = os.getcwd()
	word_path = path + "\\" + "Chinese_Embedding.txt"
	print ("word_path",word_path)
	dic, word_embedding = readChinsesWordEmbedding(word_path)

	data = []
	data_ner = []
	datasetpath = path + "\\" + "chinese_data"
	filenameList = os.listdir(datasetpath)
	filenameList.sort(reverse=True) 
	for fn in filenameList:
		if fn.find("-3.txt") > 0:
			fn = datasetpath + "\\" + fn 
			ans = splitChineseSequence(fn, dic)
			data.append(ans)

		elif fn.find("-1.txt") > 0:
			fn = datasetpath + "\\" + fn 
			data_ner.append(mapChineseNER(fn, dic, ans))

		else:
			pass
	return (data, data_ner, word_embedding)


def loadEnglishiData():
	path = os.getcwd()
	word_path = os.path.join(path, "word_embedding_english.txt")
	print('word_path', word_path)
	dic, word_embedding = readEnglishWordEmbedding(word_path)

	data = []
	data_ner = []
	datasetpath = os.path.join(path, "english_data")
	filenameList = os.listdir(path)
	filenameList.sort(reverse=True)
	for fn in filenameList:
		if fn.find("-3.txt") > 0:
			fn = os.path.join(datasetpath, fn)
			ans = splitEnglishSequence(fn, dic)
			data.append(ans)
		elif fn.find("-1.txt") > 0:
			fn = os.path.join(datasetpath, fn)
			data_ner.append(mapEnglishNER(fn, map_NER, ans))
		else:
			pass
	return (data, data_ner, word_embedding)



def DivideDataSet(dataset, nerdata):
	batch_size = len(dataset)
	train_set = []
	test_set = []
	train_lb = []
	test_lb = []
	n = random.randint(2, 7)
	dataset = np.array(dataset)
	nerdata = np.array(nerdata)
	for i in range(batch_size):
		batch = dataset[i,:,:]
		batch_ner = nerdata[i,:,:]
		batch = np.reshape(batch, (-1, max_sequence_length))
		batch_ner = np.reshape(batch_ner, (-1, max_sequence_length))
		if i % n == 0:
			test_set.append(batch)
			test_lb.append(batch_ner)
		else:
			train_set.append(batch)
			test_lb.append(batch_ner)
	pass
	return (train_set, train_lb, test_set, test_lb)


#--------------------Test functions-----------------

if __name__ == '__main__':
	print ("util.py main funcion")
	#(data, data_ner, word_embedding) = loadChineseData()  
	#print (data)
	(data, data_ner, word_embedding) = loadEnglishiData()
	print (data)
	time.sleep(10)


