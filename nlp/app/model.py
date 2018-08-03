"""
判断句子顺序的模型
"""

from nlp.content.context import Context
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda


class Layer(object):
    """
    <Embedding>
    <Context>
    <LSTM>
    <Dense>
    <Dense>

    """

    def __init__(self, tokenizer, embedding):

        pass

    def __call__(self, input, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # 先按照下表切割向量
        # 然后对词进行tokenizer

        # def slice(x, h1, h2, w1, w2):
        #     """ Define a tensor slice function
        #     """
        #     return x[:, h1:h2, w1:w2, :]
        #
        # slice_1 = Lambda(slice, arguments={'h1': 0, 'h2': 6, 'w1': 0, 'w2': 6})(sliced)
        #
        #
        # for word, i in word_index.items():
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         # words not found in embedding index will be all-zeros.
        #         embedding_matrix[i] = embedding_vector
        #
        # embedding_layer = Embedding(len(word_index) + 1,
        #                             EMBEDDING_DIM,
        #                             weights=[embedding_matrix],
        #                             input_length=MAX_SEQUENCE_LENGTH,
        #                             trainable=False)
        #
        #
        # embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        #
        # embedding = Embedding(
        # )
        pass


if __name__ == '__main__':

    # 先测试句子分割为多个向量

    test_data = []
    with open("data1.dat", 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if '' == line:
                continue
            test_data.append(line)

    # 先手动切割
    dep_set = set()

    def slice(x: str):
        global dep_set
        items = x.split("/")
        dep = items[0::3]
        op1 = items[1::3]
        op2 = items[2::3]
        # tokenizer
        print(dep)
        print(op1)
        print(op2)

        # embedding

        return dep, op1, op2

    split_data = [slice(l) for l in test_data]

    # 每个分行符是一个批次的分割

    # 生成标记
    #   方案1, 全排列
    #   方案2, 按照批次大小, 生成一个目标样本数, 然后随机生成行数的两个数, 生成样本

    # 句子距离
    #   方案1, 二元值, 有多大可能在前面还是后面
    #   方案2, 距离度量

    # embedding 方案
    #   先padding, 在运行时转换
    #   先转换, 在padding, 在输入模型

    print(dep_set)

    # 手动padding


    # print(split_data)





    # slice_1 = Lambda(slice, arguments={'h1': 0, 'h2': 6, 'w1': 0, 'w2': 6})(sliced)



    pass
