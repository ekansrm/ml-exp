"""
词向量嵌入接口类
"""


class AbstractEmbeddingBuilder(object):
    # 保存文件

    def build(self, vocab: list, cache: str = None) -> list:
        """
        词向量嵌入构建
        :return:
        """

        raise NotImplementedError('method "build" not implemented')
