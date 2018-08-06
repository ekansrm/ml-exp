"""
判断句子顺序的模型
"""

from nlp.content.context import Context
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Concatenate
from nlp.embedding.SimpleEmbeddingBuilder import SimpleEmbeddingBuilder


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

        # 假设到了Embedding成, 已经又六组数据
        # 设数据的batch_size, 句子经过padding的长度是len
        # op1: [batch_size, len], str
        # wordA1: [batch_size, len], int
        # wordB1: [batch_size, len], int

        # op2: [batch_size, len]
        # wordA2: [batch_size, len], int
        # wordB2: [batch_size, len], int

        # 经过embedding和操作子转换

        # op1: [batch_size, len], int
        # wordA1: [batch_size, len, embedding_vec], int
        # wordB1: [batch_size, len, embedding_vec], int

        # op2: [batch_size, len], int
        # wordA2: [batch_size, len, embedding_vec], int
        # wordB2: [batch_size, len, embedding_vec], int

        batch_size = 32
        max_len = 50

        op1 = Input(shape=(None, 50), dtype='int')
        wordA1 = Input(shape=(None, 50), dtype='int')
        wordB1 = Input(shape=(None, 50), dtype='int')
        op2 = Input(shape=(None, 50), dtype='int')
        wordA2 = Input(shape=(None, 50), dtype='int')
        wordB2 = Input(shape=(None, 50), dtype='int')


        # 将4个word合并, 然后通过一个embedding, 转为向量后, 再拆分

        vocab = ['我', '是', '国家']

        embedding_builder = SimpleEmbeddingBuilder('E:\BaiduNetdiskDownload\sgns.sogou.word\sgns.sogou.word')

        embedding, tokenizer = embedding_builder.build(vocab, cache='my_embedding')

        embedding_vocab = len(embedding)

        embedding_vec = len(embedding[0])

        embedding_layer = Embedding(embedding_vocab,
                                    embedding_vec,
                                    weights=[embedding],
                                    input_length=max_len,
                                    trainable=False)

        wordA1_vec = embedding_layer(wordA1)
        wordB1_vec = embedding_layer(wordB1)
        wordA2_vec = embedding_layer(wordA2)
        wordB2_vec = embedding_layer(wordB2)


        # 需要两两合并
        layer_concatenate_word1 = Concatenate(axis=[-1])

        word1 = layer_concatenate_word1([wordA1_vec, wordB1_vec])

        word2 = layer_concatenate_word1([wordA2_vec, wordB2_vec])

        context_layer = Context(
            units=100
            , contexts=4
            , activation='relu'
            , kernel_initializer='lecun_uniform'
        )

        vec1 = context_layer([op1, word1])
        vec2 = context_layer([op2, word2])

        lstm_1 = LSTM(units=1000, **kwargs)


        lstm_2 = LSTM(units=1000, **kwargs)

        lstm_1_out = lstm_1(vec1)
        lstm_2_out = lstm_2(vec2)

        final_con_layer = Concatenate(axis=[-1])

        final_con = final_con_layer([lstm_1_out, lstm_2_out])


        d_out = Dense(units=100)(final_con)
        d_out = Dense(units=2)


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
