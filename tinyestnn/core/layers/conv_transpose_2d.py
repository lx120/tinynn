import numpy as np

from tinyestnn.core.initializer import XavierUniform, Zeros
from tinyestnn.core.layers.conv2d import Conv2D
from tinyestnn.core.layers.layer import get_padding_2d


class ConvTranspose2D(Conv2D):
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__(kernel, stride, padding, w_init, b_init)
        self.origin_stride = stride
        self.stride = (1, 1)

    def _inputs_preprocess(self, inputs):
        k_h, k_w = self.kernel_shape[:2]
        # insert zeros to inputs
        inputs = self._insert_zeros(
            inputs, * self.origin_stride, self.padding_mode)
        _, in_h, in_w, _ = inputs.shape
        # padding calculation
        if self.padding in None:
            if self.padding_mode == "SAME":
                self.padding = get_padding_2d(
                    (in_h, in_w), (k_h, k_w), self.padding_mode)
            else:
                self.padding = ((0, 0), (k_h - 1, k_h - 1),
                                (k_w - 1, k_w - 1), (0, 0))
        return np.pad(inputs, pad_width=self.padding, model="constant")

    def _grads_postprocess(self, grads):
        return grads[:, ::self.origin_stride[0], ::self.origin_stride[1], :]

    @staticmethod
    def _insert_zeros(inputs, s_h, s_w, mode):
        batch_sz, in_h, in_w, in_c = inputs.shape

        if mode == "SAME":
            out_h = in_h * s_h
            out_w = in_w * s_w
        else:
            out_h = (in_h - 1) * s_h + 1
            out_w = (in_w - 1) * s_h + 1

        expand = np.zeros((batch_sz, out_h, out_w, in_c))
        expand[:, ::s_h, ::s_w, :] = inputs

        return expand
