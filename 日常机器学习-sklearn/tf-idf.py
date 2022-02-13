import numpy as np
import pandas as pd

docA = "the cat sat on my bed"
docB = "the dog sat on my knees"

bowA = docA.split(" ")
bowB = docB.split(" ")

wordSet = set(bowA).union(bowB)

print(wordSet)

# 进行词频统计

wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)

for word in bowA:
    wordDictA[word] += 1

for word in bowB:
    wordDictB[word] += 1

print(wordDictA,wordDictB)
print(pd.DataFrame( [wordDictA,wordDictB] ))

# 计算词频 TF
def computeTF(wordDict, bow):
    # 用一个字典对象记录tf, 把所有
    tfDict = {}
    nbowCount = len(bow)

    for word,count in wordDict.items():
        tfDict[word] = count / nbowCount

    return tfDict

tfA = computeTF( wordDictA, bowA)
tfB = computeTF( wordDictB, bowB)

print(tfA, '\n', tfB)

def computeIDF( wordDictList ):
    # 用一个字典保存idf
    idfDict = dict.fromkeys(wordDictList[0], 0)
    N = len(wordDictList)
    import math

    for wordDict in wordDictList:
    # 遍历字典
        for word,count in wordDict.items():
            if count > 0:
                idfDict[word] += 1

    for word, ni in idfDict.items():
        idfDict[word] =math.log10( (N + 1) / (ni + 1) )

    return idfDict


idfs = computeIDF([wordDictA,wordDictB])

print(idfs)

# 计算 tf-idf

def computeTFIDF(tf, idfs):
    tfidf = {}
    for word, tfval in tf.items():
        tfidf[word] = tfval * idfs[word]

    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

print(pd.DataFrame([tfidfA, tfidfB]))
