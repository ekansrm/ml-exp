from nlp.app.utils import *
from nlp.embedding.SimpleEmbeddingBuilder import SimpleEmbeddingBuilder
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from keras.layers import Input
from keras.models import Model

from nlp.app.model import Layer


if __name__ == '__main__':
    """
    应用测试
    """
    corpus_path = '/mnt/WORK/Project.DataScience/ml-exp/nlp/app/data1.dat'
    vocab_path = '/mnt/WORK/Project.DataScience/ml-exp/nlp/app/vocab.dat'
    corpus = read_corpus(corpus_path)

    batch_size = 32
    max_len = 50

    corpus_batches = [corpus_to_sample(batch, batch_size, score_binary) for batch in corpus]

    a_dep = []
    a_w1 = []
    a_w2 = []
    b_dep = []
    b_w1 = []
    b_w2 = []
    y = []

    vocab = read_vocab(vocab_path)

    for batch in corpus_batches:
        _dep_a, _op1_a, _op2_a, _dep_b, _op1_b, _op2_b, _y = process_sample(batch)

        a_dep.extend(_dep_a)
        a_w1.extend(_op1_a)
        a_w2.extend(_op2_a)

        b_dep.extend(_dep_b)
        b_w1.extend(_op1_b)
        b_w2.extend(_op2_b)

        y.extend(_y)

    # tokenizer
    # embedding_builder = SimpleEmbeddingBuilder('E:\BaiduNetdiskDownload\sgns.sogou.word\sgns.sogou.word')
    embedding_builder = SimpleEmbeddingBuilder('/mnt/DATA/DataScience/word2vec/sgns.sogou.word')
    embedding, tokenizer = embedding_builder.build(vocab, cache='embedding-1')
    print("词向量不存在的词数目: %d " % (len(vocab) + 1 - len(embedding)))

    a_dep = np.array([[dep_tokenizer(word) for word in sentence] for sentence in a_dep])
    a_w1 = np.array([[tokenizer(word) for word in sentence] for sentence in a_w1])
    a_w2 = np.array([[tokenizer(word) for word in sentence] for sentence in a_w2])

    b_dep = np.array([[dep_tokenizer(word) for word in sentence] for sentence in b_dep])
    b_w1 = np.array([[tokenizer(word) for word in sentence] for sentence in b_w1])
    b_w2 = np.array([[tokenizer(word) for word in sentence] for sentence in b_w2])

    padding = 'post'
    a_dep = pad_sequences(a_dep, maxlen=max_len, padding=padding)
    a_w1 = pad_sequences(a_w1, maxlen=max_len, padding=padding)
    a_w2 = pad_sequences(a_w2, maxlen=max_len, padding=padding)

    b_dep = pad_sequences(b_dep, maxlen=max_len, padding=padding)
    b_w1 = pad_sequences(b_w1, maxlen=max_len, padding=padding)
    b_w2 = pad_sequences(b_w2, maxlen=max_len, padding=padding)

    ####################################################################################################################

    op1 = Input(shape=(max_len, 1,), dtype='int32', name='a_dep')
    wordA1 = Input(shape=(max_len,), dtype='int32', name='a_w1')
    wordB1 = Input(shape=(max_len,), dtype='int32', name='a_w2')
    op2 = Input(shape=(max_len, 1,), dtype='int32', name='b_dep')
    wordA2 = Input(shape=(max_len,), dtype='int32', name='b_w1')
    wordB2 = Input(shape=(max_len,), dtype='int32', name='b_w2')

    layer = Layer(name="context")

    d_out = layer([op1, wordA1, wordB1, op2, wordA2, wordB2], embedding=embedding, name='y')

    model = Model(inputs=[op1, wordA1, wordB1, op2, wordA2, wordB2], outputs=[d_out])

    model.summary()

    model.compile(
        optimizer='adam',
        loss={'y': 'binary_crossentropy'},
        metrics=['accuracy']
    )

    ####################################################################################################################

    a_dep = np.array(a_dep).reshape([-1, max_len, 1])
    a_w1 = np.array(a_w1)
    a_w2 = np.array(a_w2)

    b_dep = np.array(b_dep).reshape([-1, max_len, 1])
    b_w1 = np.array(b_w1)
    b_w2 = np.array(b_w2)

    y = np.array(y)

    y_df = pd.DataFrame({'y': y})
    y = pd.get_dummies(y_df, columns=['y']).values

    ####################################################################################################################

    model.fit(
        x={'a_dep': a_dep, 'a_w1': a_w1, 'a_w2': a_w2,
           'b_dep': b_dep, 'b_w1': b_w1, 'b_w2': b_w2, },
        y={'y': y},
        epochs=50, batch_size=32, shuffle=True, validation_split=0.2,
        verbose=1,
    )




