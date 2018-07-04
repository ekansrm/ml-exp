"""
Simple Word Embedding Builder
简单词向量嵌入生成和管理

- 对于embedding文件, 生成词的偏移量
- 抽取的词向量进行缓存, 检查词向量是否有更新, 有则重新加载

"""
import os
from typing import Union, Optional, Any, Dict, Tuple

from nlp.embedding.SimpleEmbeddingLookup import SimpleEmbeddingLookup
from nlp.embedding.Utils import save_var, load_var


class SimpleEmbeddingBuilder(object):

    def __init__(self, embedding: str):
        self._embedding = SimpleEmbeddingLookup(embedding)
        pass

    def build(self, vocab: list, cache: str = None) -> Tuple[Union[Optional[list], Any], Union[Optional[Dict[Any, int]], Any]]:

        def _build_token(m_vocab):
            m_token = {}
            for i, word in enumerate(m_vocab):
                m_token[word] = i
            return m_token

        embedding = None
        token = None
        renew = False

        if cache is not None:
            if os.path.exists(cache):
                [embedding, token] = load_var(cache)
                renew = len(set(vocab) - set(token.keys())) > 0

        if renew or cache is None:
            embedding = list([self._embedding[word] for word in vocab])
            token = _build_token(vocab)

        if cache is not None:
            save_var([embedding, token], cache)

        def tokenizer(word) -> int:
            return token[word]

        return embedding, tokenizer


if __name__ == '__main__':
    vocab = ['of', 'the', 'he', 'she']

    embedding_builder = SimpleEmbeddingBuilder('embedding.txt')

    embedding, tokenizer = embedding_builder.build(vocab)

    print(embedding_builder)
    print(list([tokenizer(word) for word in vocab]))
