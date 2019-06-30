import tensorflow as tf
import libs.bnn.weight_blocks as wb
from collections import OrderedDict

# Names for various approximations that can be requested for weight matrix.
APPROX_KRONECKER_NAME = "kron"
APPROX_BBB_NAME = "bbb"

_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: wb.MVGBlock,
    APPROX_BBB_NAME: wb.BBBBlock,
}


class sampler(object):
    def __init__(self, coeff, ita, particles, approx):
        self._coeff = coeff
        self._ita = ita
        self._particles = particles
        self._approx = approx
        self.blocks = OrderedDict()

    def get_block(self):
        return self.blocks.values()

    def get_params(self, idx):
        if idx not in self.blocks:
            raise ValueError("Unvalid query: {}".format(idx))
        return self.blocks[idx].params()

    def sample(self, idx, mode):
        if idx not in self.blocks:
            raise ValueError("Unvalid query: {}".format(idx))
        return self.blocks[idx].sample(mode)

    def register_block(self, idx, shape):
        if idx in self.blocks:
          raise ValueError("Duplicate registration: {}".format(idx))

        block_type = _APPROX_TO_BLOCK_TYPES[self._approx]
        self.blocks[idx] = block_type(idx, shape, self._coeff, self._ita, self._particles)

    def kl_div(self):
        kl = 0.
        for wb in self.blocks.values():
            kl += wb.kl_div()
        return kl

    def update(self, blocks):
        if self._approx == APPROX_BBB_NAME:
            raise ValueError("Don't support update function in BBB")
        block_list = self.get_block()
        update_op = [wb.update(fb) for wb, fb in zip(block_list, blocks)]
        return tf.group(*update_op)

    def update_randomness(self):
        block_list = self.get_block()
        update_op = [wb.update_randomness() for wb in block_list]

        return tf.group(*update_op)

    def compute_vFv(self, vecs):
        trans_vecs = []
        for (idx, wb), vec in zip(self.blocks.items(), vecs):
            trans_vecs.append(wb.multiply(vec))

        # terms = [tf.reduce_sum(vec * vec) for vec in trans_vecs]
        # return tf.reduce_sum(terms)
        terms = tf.convert_to_tensor([tf.reduce_sum(vec * vec, 1) for vec in trans_vecs])
        return tf.reduce_sum(terms, 0)
