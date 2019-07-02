import numpy as np
import jieba

#分词
def sent2word(line):
	segList = jieba.lcut_for_search(line, HMM=False)
	return segList

def singEmbed(embeddings_index, word):
	if word in embeddings_index:
		return embeddings_index[word]
	else:
		return np.zeros(300)

#center word 和 outside words 的词向量
def getWordVec(wordseq, wordindex, embeddings_index, windowsize=5):
	#wordindex是center word 的位置
	tempword = wordseq[wordindex]

	wordVec = np.zeros([5,300])

	if tempword in embeddings_index:
		wordVec[0] = embeddings_index[tempword]
		loc = wordindex
		vecIn = 1
		windowsize = min(windowsize, len(wordseq))
		if wordindex < 2:
			for ii in range(loc):
				tempword = wordseq[ii]
				wordVec[vecIn] = singEmbed(embeddings_index, tempword)
				vecIn += 1
			for ii in range(loc+1, windowsize):
				tempword = wordseq[ii]
				wordVec[vecIn] = singEmbed(embeddings_index, tempword)
				vecIn += 1
		elif wordindex > len(wordseq)-3:
			for ii in range(len(wordseq)-windowsize, loc):
				tempword = wordseq[ii]
				wordVec[vecIn] = singEmbed(embeddings_index, tempword)
				vecIn += 1
			for ii in range(loc+1, len(wordseq)):
				tempword = wordseq[ii]
				wordVec[vecIn] = singEmbed(embeddings_index, tempword)
				vecIn += 1
		else:
			for ii in range(1,2):
				tempword1 = wordseq[loc-ii]
				tempword2 = wordseq[loc+ii]
				wordVec[3-ii] = singEmbed(embeddings_index, tempword1)
				wordVec[3+ii] = singEmbed(embeddings_index, tempword2)
	return wordVec

#计算内积之和
def calculProb(wordVec, windowsize=5):
	y = np.zeros(windowsize-1)
	for i in range(1, windowsize):
		y[i-1] = np.dot(wordVec[0], wordVec[i])

	return y.sum()

#sentence score
def SentScore(embeddings_index, sent1=""):
	wordseq1=sent2word(sent1)
	y1 = 0
	num1 = len(wordseq1)
	for tmpIn in range(0, num1):
		wordVec1=getWordVec(wordseq1, tmpIn, embeddings_index)
		y1 += calculProb(wordVec1)
	y1 /= num1
	#print('y1: ', y1)
	return y1

def getEmbed(file):
	embeddings_index = {}
	f = open(file,encoding='utf-8')
	count_num = 0
	#for line in f:
	#	break
	for line in f:
		if line[0]>='0':
			if line[0]<='9':
				continue
		if line[0]>='a':
			if line[0]<='z':
				continue
		if line[0]>='A':
			if line[0]<='Z':
				continue
		if line[1]=='\u3000':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
		
		count_num+=1
		if count_num>50000:
			break
	f.close()
	return embeddings_index
