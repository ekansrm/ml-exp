import pickle


def save_var(var, path):
    """
    保存变量至文件
    :param var: 变量
    :param path: 保存路径
    :return:
    """
    with open(file=path, mode="wb") as fp:
        pickle.dump(var, file=fp)


def load_var(path):
    """
    从文件读取变量值
    :param path: 保存路径
    :return:
    """
    with open(file=path, mode="rb") as fp:
        return pickle.load(fp)
