import os
import logging
from tqdm import tqdm

from nlp.embedding import Utils
from typing import List, Dict, Tuple, TextIO


class Lazy(object):
    """
    延迟加载
    """
    def __init__(self, func):
        self.func = func  # lazy的实例属性func等于area方法对象

    def __get__(self, instance, cls):
        val = self.func(instance)                   # 调用area方法计算出结果
        setattr(instance, self.func.__name__, val)  # 将结果设置给c的area属性
        return val


logger = logging.getLogger("SimpleEmbeddingLookup")


class SimpleEmbeddingLookup(object):

    """
    Embedding 文件
    """

    def __init__(self, embedding_path: str):
        self._embedding_path = os.path.abspath(embedding_path)

        self._embedding_fp = None
        self._embedding_index = None
        self._embedding_cache = {}

    def __del__(self):
        """
        结束时要关闭文件
        :return:
        """
        if self._embedding_fp:
            self._embedding_fp.close()
            logger.info('close embedding file...')

    @Lazy
    def embedding_fp(self) -> TextIO:
        logger.info('open embedding file...')
        if not self._embedding_fp:
            self._embedding_fp = open(self._embedding_path, 'r', encoding='utf-8')
        return self._embedding_fp

    @Lazy
    def embedding_index(self) -> Dict[str, Tuple[int, int]]:
        logger.info("initialize embedding index")
        if not self._embedding_index:
            embedding_index_path = self._embedding_path + ".index"
            if not os.path.exists(embedding_index_path):
                logger.info("embedding index does not exist")
                logger.info("indexing embedding...")
                index = self._build_embedding_index()
                self._embedding_index = index
                logger.info("index embedding completed.")
                Utils.save_var(self._embedding_index, embedding_index_path)
            else:
                logger.info("embedding index exists")
                logger.info("loading embedding index...")
                index = Utils.load_var(embedding_index_path)
                logger.info("load embedding index completed")
                self._embedding_index = index

        return self._embedding_index

    def _build_embedding_index(self) -> Dict[str, Tuple[int, int]]:

        embedding_fp = self.embedding_fp

        embedding_index = {}
        line_byte_cnt = 0
        word = ''
        word_completed = False
        embedding_fp.seek(0, 2)
        embedding_file_size = embedding_fp.tell()

        embedding_fp.seek(0, 0)
        bar_format = "{percentage:3.0f}% | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}]"
        with tqdm(total=embedding_file_size, bar_format=bar_format) as p_bar:
            last_pos = 0
            while True:
                c = embedding_fp.read(1)
                is_line_break = False

                # 如果不是最后一个字符
                if c not in ['', '\n', '\r', '\r\n']:
                    line_byte_cnt += 1
                else:
                    p_bar.update(embedding_fp.tell() - last_pos)
                    last_pos = embedding_fp.tell()
                    is_line_break = True

                # word为当前行的第一个空格前的字符串
                if not is_line_break and not word_completed:
                    if c == " ":
                        word_completed = True
                    else:
                        word = word + c

                # 如果已经是最后一个字符, 或者遇到换行符, 当前行结束. 写入词和当前行的偏移量和长度
                if is_line_break and line_byte_cnt is not 0:
                    embedding_index[word] = (embedding_fp.tell() - line_byte_cnt - len(c), line_byte_cnt)
                    line_byte_cnt = 0
                    word_completed = False
                    word = ''

                # 如果读到最后一个字符, embedding 读取和索引结束
                if c is '':
                    break

        return embedding_index

    def _get_vector(self, word: str) -> List[float]:
        embedding_fp = self.embedding_fp
        pos = self.embedding_index[word]
        embedding_fp.seek(pos[0])
        line = embedding_fp.read(pos[1])
        items = line.split(" ")
        word_in_line = items[0]
        if word_in_line != word:
            raise Exception('index for word "%s" is not correct, abort' % word)
        return list(map(float, items[1:]))

    def __getitem__(self, word) -> List[float]:
        if word not in self._embedding_cache:
            vec = self._get_vector(word)
            self._embedding_cache[word] = vec
        else:
            vec = self._embedding_cache[word]

        return vec


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG
    )
    embedding = SimpleEmbeddingLookup('D:\Work\Project\Project.DataScience\model\glove\glove.840B.300d.txt')
    print(embedding['the'])
    print(embedding['is'])
    print(embedding['a'])
    print(embedding['king'])
