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
    with open("data1.dat", 'r') as fp:
        for line in fp.readlines():
            if '' == line:
                continue
            test_data.append(line)
    print(test_data)

    # def slice(x, h1, h2, w1, w2):
    #     """ Define a tensor slice function
    #     """
    #     return x[:, h1:h2, w1:w2, :]
    #
    # slice_1 = Lambda(slice, arguments={'h1': 0, 'h2': 6, 'w1': 0, 'w2': 6})(sliced)

    # 先手动切割

    def slice(x: str):
        items = x.split("/")
        return items[0::3], items[1::3], items[2::3]

    split_data = [slice(l) for l in test_data]

    print(split_data)





    # slice_1 = Lambda(slice, arguments={'h1': 0, 'h2': 6, 'w1': 0, 'w2': 6})(sliced)



    pass
