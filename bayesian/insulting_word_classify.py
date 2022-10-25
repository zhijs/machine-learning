


from typing import List, Tuple
from numpy import *
import numpy as np

def create_vocb_list(list_post: matrix)-> matrix:
    """
    Desc:
      获取所有单词集合
    Args:
      list_post -- 数据集
    Returns:
      list -- 所有单词的集合，不含重复的元素的单词列表
    """
    vocab_set = set()
    for item in list_post:
        # | 取并集,适用于与 set, eg set([1,2]) | set([3, 4]) -> set([1, 2, 3, 4])
        vocab_set = vocab_set | set(item)
    return list(vocab_set)

def set_of_word2vec(vocab_list: List, input_set: matrix)-> matrix:
    """
    Desc:
      遍历查看改单词是否出现，出现改单词则将改单词置1
    Args:
      vocab_list -- 所有的单词和合集
      input_set -- 输入的数据集
    """
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            print('the word :{} is not in my vocabulary'.format(word))
            pass
    return result

def load_data_set()-> Tuple[matrix, matrix]:
    """
    Desc: 
      创建数据集
    Return:
      posting_list -- 单词列表(1 是有侮辱性的)
      class_vec -- 所属类别
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec

def train_naive_bayes(tran_mat: matrix, list_classes: matrix)-> tuple([matrix, matrix, float]):
    """
    Desc:
      朴素贝叶斯分类
    Args:
      tran_mat -- 输入的数据文本
      list_classes -- 文本数据对应的类别
    Return:
      p1vec -- 每个单词在侮辱性类别中出现的概率
      p0vec -- 每个单词在非侮辱性类别中出现的概率
      pos_abusive -- 侮辱性类别出现的概率
    """
    train_doc_num = len(tran_mat)
    word_num = len(tran_mat[0])
    
    # 侮辱性类别所占的概率
    pos_abusive = np.sum(list_classes)/train_doc_num
    
    # 存储每个单词，出现在非侮辱性类别中的个数
    p0num = np.ones(word_num)

    # 存储每个单词，出现在侮辱性类别中的个数
    p1num = np.ones(word_num)

    # 整个数据集单词出现的次数
    p0num_all = 0
    p1num_all = 0

    for i in range(train_doc_num):
        # 遍历文件，如果是侮辱性文件，就计算侮辱性文件中出现的单词个数
        if list_classes[i] == 1:
            p1num += tran_mat[i]
            p1num_all += np.sum(tran_mat[i])
        else:
            p0num += tran_mat[i]
            p0num_all += np.sum(tran_mat[i])
    
    p1vec = p1num / p1num_all
    p0vec = p0num / p0num_all

    return p0vec, p1vec, pos_abusive


def classify_naive_bayes(vec2classify: matrix, p0vec: matrix, p1vec: matrix, p_class1: float):
    """
    Desc:
      使用朴素贝叶斯公式计算
    Args:
      vec2classify -- 带测试的数据向量
      p0vec -- 每个单词在非侮辱性类别中出现的概率向量
      p1vec -- 每个单词在侮辱性类别中出现的概率向量
      p_class1 -- 侮辱性类别的概率
    """
    # 计算测试数据属于非侮辱性类别的概率
    p1 = np.sum(vec2classify * p1vec) * p_class1
    p0 = np.sum(vec2classify * p0vec) * (1 - p_class1)
    if p1 > p0:
      return 1
    else:
      return 0



def testing_naive_bayes():
    """
    Desc:
      测试朴素贝叶斯方法
    Returns:
      none
    """
    # 加载数据
    list_post, list_classes = load_data_set()

    # 创建单词集合
    vocab_list = create_vocb_list(list_post)

    train_mat = []
    for post_in in list_post:
      train_mat.append(
        set_of_word2vec(vocab_list, post_in)  
      )
    # 训练数据
    p0v, p1v, p_abusive = train_naive_bayes(np.array(train_mat), np.array(list_classes))
    # 测试数据
    test_one = ['love', 'my', 'dalmation']
    test_one_doc = np.array(set_of_word2vec(vocab_list, test_one))
    print('the result is: {}'.format(classify_naive_bayes(test_one_doc, p0v, p1v, p_abusive)))
    # 计算
    test_two = ['stupid', 'garbage']
    test_two_doc = np.array(set_of_word2vec(vocab_list, test_two))
    print('the result is: {}'.format(classify_naive_bayes(test_two_doc, p0v, p1v, p_abusive)))


if __name__ == '__main__':
    testing_naive_bayes()