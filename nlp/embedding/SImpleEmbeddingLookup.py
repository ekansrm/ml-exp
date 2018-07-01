from tqdm import tqdm
import os

from nlp.embedding import Utils


class SimpleEmbeddingLookup(object):
    """
    lazy load
    """

    def __init__(self, embedding_path: str):
        self._embedding_path = embedding_path

        self._embedding_fp = None
        self._embedding_index = None
        self._embedding_cache = {}

        self._build_embedding_index()

    def __del__(self):
        """
        结束时要关闭文件
        :return:
        """
        if self._embedding_fp:
            self._embedding_fp.close()

    def _get_embedding_fp(self):

        if not self._embedding_fp:
            self._embedding_fp = open(self._embedding_path, 'r', encoding='utf-8')
        return self._embedding_fp

    def _get_embedding_index(self):
        if not self._embedding_index:
            embedding_index_path = self._embedding_path + ".index"
            if not os.path.exists(embedding_index_path):
                self._build_embedding_index()
                Utils.save_var(self._embedding_index, embedding_index_path)
            else:
                index = Utils.load_var(embedding_index_path)
                self._embedding_index = index

        return self._build_embedding_index()

    def _build_embedding_index(self):

        embedding_fp = self._get_embedding_fp()

        index = {}
        linecnt = 0
        word = ''
        getword = False
        embedding_fp.seek(0, 2)
        file_size = embedding_fp.tell()
        embedding_fp.seek(0, 0)
        bar_format = "{percentage:3.0f}% | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}]"
        with tqdm(total=file_size, bar_format=bar_format) as p_bar:
            while True:
                c = embedding_fp.read(1)

                # 如果不是最后一个字符
                if c is not "":
                    linecnt += 1

                # word为第一个空格前的字符
                if getword is False:
                    if c == " ":
                        getword = True
                    else:
                        word = word + c

                # 如果已经是最后一个字符, 或者遇到回车符
                if c == '\n' or c is "":
                    if linecnt is not 0:
                        index[word] = (embedding_fp.tell() - linecnt, linecnt)
                    p_bar.update(linecnt)
                    linecnt = 0
                    getword = False
                    word = ''

                if c is "":
                    break
        self._embedding_index = index

    def _get_vec(self, word):

        pos = self._embedding_index[word]
        self._embedding_fp.seek(pos[0])
        line = self._embedding_fp.read(pos[1])
        items = line.split(" ")
        word_in_embedding_file = items[0]
        if word_in_embedding_file != word:
            raise Exception("index not correct")
        return list(map(float, items[1:]))

    def __getitem__(self, word):
        if word not in self._embedding_cache:
            vec = self._get_vec(word)
            self._embedding_cache[word] = vec
        else:
            vec = self._embedding_cache[word]

        return vec


if __name__ == '__main__':
    embedding = SimpleEmbeddingLookup('/mnt/WORK/Project.DataScience/ml-exp/embedding.txt')
    print(embedding['of'])
    print(embedding['the'])
