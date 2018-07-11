import numpy as np
import time
from nlp.embedding.Utils import save_var, load_var


class OpDataGenerator(object):

    def __init__(self, delta, category, DIM):
        self.delta = delta
        self.category = category
        self.DIM = DIM

        self.cate_list = np.array([np.random.permutation(np.arange(0, 1, 1 / DIM)) for i in range(0, category, 1)])

        # np.random.seed(time.time())

        def op_add(vec):
            op_1 = np.random.normal(loc=0., scale=delta, size=(DIM,))
            op_2 = vec - op_1
            return op_1, op_2

        def op_sub(vec):
            op_1 = np.random.normal(loc=0., scale=delta, size=(DIM,))
            op_2 = vec + op_1
            return op_1, op_2

        def op_mul(vec):
            op_1 = np.random.normal(loc=0., scale=delta, size=(DIM,))
            op_2 = vec / op_1
            return op_1, op_2

        def op_div(vec):
            op_1 = np.random.normal(loc=0., scale=delta, size=(DIM,))
            op_2 = vec * op_1
            return op_1, op_2

        self.op_table = {
            0: op_add,
            1: op_sub,
            2: op_mul,
            3: op_div,
        }
        self.OP_CA = len(self.op_table)

    def _gen_one_sample(self):
        category_idx = np.random.randint(low=0, high=self.category, size=None)
        delta_rand = np.random.normal(loc=0., scale=self.delta, size=(self.DIM,))
        core = self.cate_list[category_idx] + delta_rand
        op_idx = np.random.randint(low=0, high=self.OP_CA, size=None)
        op1, op2 = self.op_table[op_idx](core)
        return op_idx, op1, op2, category_idx

    def gen(self, num):
        op_idx_a = []
        op_1_a = []
        op_2_a = []
        category_a = []
        for i in range(num):
            op_idx, op1, op2, category = self._gen_one_sample()
            op_idx_a.append(op_idx)
            op_1_a.append(op1)
            op_2_a.append(op2)
            category_a.append(category)
        op_idx_a = np.array(op_idx_a)
        op_1_a = np.array(op_1_a)
        op_2_a = np.array(op_2_a)
        category_a = np.array(category_a)

        return op_idx_a, op_1_a, op_2_a, category_a


if __name__=='__main__':
    gen = OpDataGenerator(delta=0.01, category=9, DIM=12)
    save_var(gen.gen(100000), "test_data")

    op_idx_a, op_1_a, op_2_a, category_a = load_var("test_data")






