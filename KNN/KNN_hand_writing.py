# KNN 算法实战2-手写体识别
# 导入科学计算包numpy和运算符模块operator
import string
from typing import Tuple
from numpy import *
import os
import numpy as np
from collections import Counter

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

def img2Vector(fileName: string) -> matrix:
    """
    Desc:
      将图像矩阵转化为向量 32*32 -> 1*1024
    Args:
      fileName -- 图片路径
    Return:
      returnVect -- 图片向量      
    """
    returnVect = zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
           returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
     Desc:
      手写数字识别分类器，并打印错误数和错误率
    """
    hwLabels = []
    # 加载数据
    traningFileList = os.listdir('./data/digits/trainingDigits')
    traningLen = len(traningFileList)
    tranningMat = zeros((traningLen, 1024))
    for i in range(traningLen):
        fileNameStr = traningFileList[i]
        fileStr = fileNameStr.split('.')[0] # 0_0/1_1
        classNumstr = fileStr.split('_')[0]
        hwLabels.append(classNumstr)

        # 讲 32*32 的图像矩阵转化为 1*1024 的矩阵
        tranningMat[i, :] = img2Vector('./data/digits/trainingDigits/%s' % fileNameStr)
    
    # 测试数据
    testFileList = os.listdir('./data/digits/testDigits')
    errorCount = 0
    testLen = len(testFileList)
    for i in range(testLen):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectResult = img2Vector('./data/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectResult, tranningMat, hwLabels, 3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, classNumStr))
        errorCount += classifierResult != classNumStr
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / testLen))
if __name__ == '__main__':
    handwritingClassTest()