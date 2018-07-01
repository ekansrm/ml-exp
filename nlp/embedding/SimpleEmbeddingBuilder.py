"""
Simple Word Embedding Builder
简单词向量嵌入生成和管理

- 对于embedding文件, 生成词的偏移量
- 抽取的词向量进行缓存, 检查词向量是否有更新, 有则重新加载

"""
import pickle

from .AbstractEmbeddingBuilder import AbstractEmbeddingBuilder


class SimpleEmbeddingBuilder(AbstractEmbeddingBuilder):

    def __init__(self):
        pass

    @staticmethod
    def _save_var(var, path):
        """
        保存变量至文件
        :param var: 变量
        :param path: 保存路径
        :return:
        """
        with open(file=path, mode="wb") as fp:
            pickle.dump(var, file=fp)

    @staticmethod
    def _load_var(path):
        """
        从文件读取变量值
        :param path: 保存路径
        :return:
        """
        with open(file=path, mode="rb") as fp:
            return pickle.load(fp)



    def build(self, vocab: list, cache: str = None) -> list:
        pass

    def get_tokenizer(self) -> function:
        pass
