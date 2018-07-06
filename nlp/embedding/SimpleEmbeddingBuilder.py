"""
Simple Word Embedding Builder
简单词向量嵌入生成和管理

- 对于embedding文件, 生成词的偏移量
- 抽取的词向量进行缓存, 检查词向量是否有更新, 有则重新加载

"""
import os
import logging
from typing import Tuple, List, Callable

from nlp.embedding.SimpleEmbeddingLookup import SimpleEmbeddingLookup
from nlp.embedding.Utils import save_var, load_var

logger = logging.getLogger("SimpleEmbeddingBuilder")


class SimpleEmbeddingBuilder(object):

    def __init__(self, embedding_path: str, embedding_encoding: str = 'utf-8'):
        self._embedding = SimpleEmbeddingLookup(embedding_path, embedding_encoding)
        self._embedding_name = os.path.basename(embedding_path)

    def build(self, vocabulary: List[str], cache: str = None) -> Tuple[List[List[float]], Callable[[str], int]]:

        def _build_token(m_vocab):
            m_token = {}
            for i, word in enumerate(m_vocab):
                m_token[word] = i
            return m_token

        base_embedding_name = self._embedding_name
        custom_embedding = None
        token = None
        rebuild = True

        if cache is not None:
            logger.info("load embedding cache")
            if os.path.exists(cache):
                logger.info("embedding cache found")
                (cached_basic_embedding_name, custom_embedding, token) = load_var(cache)
                rebuild = len(set(vocabulary) - set(token.keys())) > 0
                if rebuild:
                    logger.info("there is new word in vocabulary, need to rebuild custom embedding")
                if cached_basic_embedding_name != base_embedding_name:
                    rebuild = True
                    logger.info("base embedding file has changed, need to rebuild custom embedding")
            else:
                logger.warning("custom embedding cache not found")

        if rebuild:
            logger.info("building custom embedding...")
            custom_embedding = list([self._embedding[word] for word in vocabulary])
            token = _build_token(vocabulary)
            logger.warning("build custom embedding completed")

        if cache is not None:
            save_var((base_embedding_name, custom_embedding, token), cache)

        def _tokenizer(word: str) -> int:
            return token[word]

        return custom_embedding, _tokenizer


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG
    )
    vocab = ['我', '是', '国家']

    embedding_builder = SimpleEmbeddingBuilder('E:\BaiduNetdiskDownload\sgns.sogou.word\sgns.sogou.word')

    embedding, tokenizer = embedding_builder.build(vocab, cache='my_embedding')

    vocab2 = ['你', '是', '国家']
    embedding2, tokenizer2 = embedding_builder.build(vocab2, cache='my_embedding')

    print(list([tokenizer(word) for word in vocab]))
    print(list([tokenizer2(word) for word in vocab2]))
