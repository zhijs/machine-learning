#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 决策树算法实战：预测隐形眼镜类型

from ast import operator
from collections import Counter
import math
import string
from numpy import *
import numpy as np
import decisionTreePlot as dtPlot

def majorityCnt(classList: matrix):
  """
  Desc:
    选择出现次数最多的结果
  Args:
    classList -- label 列的集合
  Returns:
    bestFeature -- 最优的特征列
  """
  classCount = {}
  for vote in classList:
    if vote not in classCount.keys():
      classCount[vote] = 0
    classCount[vote] += 1
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

def splitDataSet(dataSet: matrix, index: int, value: string)-> matrix:
    """
    Desc:
      划分数据集
      splitDataSet(通过遍历 dataSet数据集，求出 index 对应的 Colnum 列的值为 value 的行 )
    Args:
      dataSet -- 数据集
      index -- 划分数据的特征所在的 index
      value --特征的值
    Return:
      index 列尾 value 的数据集 【该数据集需要排除index列】
    """
    retDataSet = []
    for featVet in dataSet:
      # 判断 index 的值是否等于 value
      if featVet[index] == value:
         # 去除 index 对应的列
         reduceFeatVec: matrix = featVet[:index]
         reduceFeatVec.extend(featVet[index+1: ])
         retDataSet.append(reduceFeatVec)
    return retDataSet


def calcShannonEnt(dataSet: matrix):
    """
    Desc:
      计算给定数据集的香农熵
    Args:
      dataSet -- 数据集
    Returns:
      shannonEnt -- 返回每一组 feature 下的某个分类下的香农熵
    """
    # 计算标签出现的次数
    label_count = Counter(data[-1] for data in dataSet)
    
    # 计算概率
    probs = [p[1] / len(dataSet) for p in label_count.items()]

    # 计算香农熵
    shannonEnt = sum([- p * math.log(p, 2) for p in probs])

    return shannonEnt

def chooseBestFeatureToSplit(dataSet: matrix):
    """
    Desc:
      选择切分数据最佳的特征
    Args:
      dataSet -- 需要切分的数据集
    Returns:
      bestFeatures -- 切分数据集最优的特征列
    """
    # 计算初始香农熵
    base_entropy = calcShannonEnt(dataSet)
    numFeatures = len(dataSet[0]) - 1
    # 最优的信息增益值，和最优的 Feature 编号
    bestEntropyGain, bestFeature = 0.0, -1

    # 遍历没个特征
    for i in range(numFeatures):
        
        # 特征集合列
        featList = [example[i] for example in dataSet]

        # 去重
        uniqueVals = set(featList)

        # 临时信息熵
        newEntropy = 0.0

        # 遍历当前特征中的所有唯一属性值，对每个唯一属性划分一次数据集，计算数据集的心熵值，
        # 并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        entropyGain = newEntropy - base_entropy
        if entropyGain > bestEntropyGain:
            bestEntropyGain = entropyGain
            bestFeature = i
    return bestFeature
        



def createTree(dataSet: matrix, labels: matrix):
    """
    Desc:
      创建决策树
    Args:
      dataSet -- 要创建决策树的训练数据集
      labels -- 训练数据集中数据的特征属性
    Returns:
      Tree -- 决策树
    """
    classList = [example[-1] for example in dataSet]
    # 只有一个类别，直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    # 第二个停止条件
    if len(dataSet[0]) == 1:
      return majorityCnt(classList)
    # 选择最优的列属性，作为分支
    bestFeatIndex = chooseBestFeatureToSplit(dataSet)

    bestFeatLabel = labels[bestFeatIndex]

    # 初始化决策树
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeatIndex])
    # 取出最优列，做分类
    featValues = [example[bestFeatIndex] for example in dataSet]
    
    uniqueValues = set(featValues)

    for value in uniqueValues:
       subLabels = labels[:]
       myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeatIndex, value), subLabels) 
  
    return myTree     






    
def contactLensesTest(): 
    """
    Desc:
        预测隐形眼镜类型
    Args:
        none
    Returns:
        none
    """
    #加载隐形眼镜相关的数据
    fr = open('/home/data/GROUP/yangzhi/project/python/machine-learning/Decison_Tree/data/lenses.txt')
    # 解析数据，获得 features
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 数据对应的属性名称
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

    # 创建决策树
    lenseesTree = createTree(lenses, lensesLabels)
    print(lenseesTree)
    dtPlot.createPlot(lensesTree)


if __name__ == "__main__":
    contactLensesTest()