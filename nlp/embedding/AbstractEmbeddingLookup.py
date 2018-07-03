"""
词向量嵌入查找表接口类
"""


class AbstractEmbeddingLookUp(object):

    def _get_vector(self, word):
        raise NotImplementedError('method "_get_vector" is not implemented')
