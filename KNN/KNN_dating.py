
# KNN 算法实战1-对约会对象进行聚类
# 导入科学计算包numpy和运算符模块operator
from typing import Tuple
from numpy import *
import os
import numpy as np
import operator
from collections import Counter

def file2Matrix(filePath):
    """
    从文件加载数据
    : param filePath 数据文件路径
    : return 数据矩阵和对应类别的 label
    """
    fr = open(filePath, 'r')
    numberOflines = len(fr.readlines())
    
    # 初始化空矩阵
    returnMat = zeros((numberOflines, 3))

    # label 向量
    classLabelVector = []
    fr = open(filePath, 'r')
  
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

def normalize(dataMat: matrix) -> Tuple[matrix,int, int]:
    """
      Desc: 
        归一化特征值，消除属性之间量级不同导致的影响
      Args:
        dataSet: 需要归一化的数据集
      Return:
        normDataSet --归一化的到的数据集
        ranges --归一化处理的范围
        minVals -- 最小值 
      归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
    """
    minVals = dataMat.min(0)
    maxVals = dataMat.max(0)
    ranges = maxVals - minVals
    normData = zeros(shape(dataMat))
    normData = (dataMat - minVals) / ranges
    return normData, ranges, minVals


def classify0(inX: matrix, dataSet: matrix, labels: matrix, k: int) -> matrix:
  """
    Desc:
      KNN 分类函数
    Args:
      inX -- 用于分类的输入向量/测试数据
      dataSet -- 训练数据集 features
      labels -- 训练数据集的 labels
      k -- 选择最近的数目
    Returns:
      sortedClassCount[0][0] -- 输入向量的预测分类 labels 

    程序使用欧式距离公式.
  """
  # 计算距离
  # 欧氏距离:  点到点之间的距离
  #    第一行:  同一个点 到 dataSet的第一个点的距离。
  #    第二行:  同一个点 到 dataSet的第二个点的距离。
  #    ...
  #    第N行:  同一个点 到 dataSet的第N个点的距离。

  # inX - dataSet eg: []
  # (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
  # np.sum([[0, 1], [0, 5]], axis=0), 每行的加和
  # np.sum([[0, 1], [0, 5]], axis=1)， 每列的加和
  # dist 为 InX 向量到每个目标距离的向量
  dist: matrix = np.sum((inX - dataSet)**2, axis=1)**0.5

  # 对 dist 做排序, 并取前 k 个
  Klabels = [labels[index] for index in dist.argsort()[0: k]]

  # 计算出现最多的类别
  label = Counter(Klabels).most_common()[0][0]

  return label

def datingClassTest():
   """
    Desc: 
        对约会网站的测试方法，并将分类错误的数量和分类错误率打印出来
    Args: 
        None
    Returns: 
        None
    """
   hoRatio = 0.1 # 测试数据的比例
   print("当前路径 -  %s" %os.getcwd())
   # 从文件加载数据
   datingDataMat, datingLabels = file2Matrix('/home/data/GROUP/yangzhi/project/python/machine-learning/KNN/data/datingTestSet2.txt')
   
   # 归一化数据
   normMat, ranges, minVals = normalize(datingDataMat)

   row = normMat.shape[0]

   # 测试样本
   numTest = int(row * hoRatio)

   print('numTestVecs=', numTest)

   errorCount = 0

   for i in range(numTest):
       classifierResult = classify0(normMat[i], normMat[numTest: row], datingLabels[numTest: row], 3)
       print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
       errorCount += classifierResult != datingLabels[i]
      
   print("the total error rate is: %f" % (errorCount / numTest))
   print(errorCount)
if __name__ == '__main__':
  datingClassTest()