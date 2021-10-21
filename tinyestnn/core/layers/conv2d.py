import numpy as np

from tinyestnn.core.initializer import XavierUniform, Zeros
from tinyestnn.core.layers.layer import Layer, get_padding_2d, im2col


class Conv2D(Layer):
    """Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (height, width,
        in_channels, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: Weight initializer
    :param b_init: Bias initializer
    """
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super.__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": self.kernel_shape, "b": self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None

        self.ctx = None

    def forward(self, inputs):
        """
        Accelerate convolution via im2col trick.
        An example (assuming only one channel and one filter):
        input = | 43 16 78 |              kernel = | 4  6 |
          (X)   | 34 76 95 |                       | 7  9 |
                | 35  8 46 |

        After im2col and kernel flattening:
         col = | 43  16  34  76 |         kernel = | 4 |
               | 16  78  76  95 |           (W)    | 6 |
               | 34  76  35   8 |                  | 7 |
               | 76  95   8  46 |                  | 9 |
        """
        if not self.is_init:
            self._init_params()

        k_h, k_w, _, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)  # pad, N * H * W * C

        # padded inputs to column matrix
        col = im2col(X, k_h, k_w, s_h, s_w)
        # perform convolution by matrix product.
        W = self.params["w"].reshape(-1, out_c)
        Z = col @ W
        # reshape output
        batch_sz, in_h, in_w, _ = X.shape
        # separate the batch size and feature map dimension
        Z = Z.reshape(batch_sz, Z.shape[0] // batch_sz, out_c)
        # further divide the feature map in to (h, w) dimension
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        Z = Z.reshape(batch_sz, out_h, out_w, out_c)

        # plus  the bias for every filter
        Z += self.params["b"]
        # save results for backward function
        self.ctx = {"X_shape": X.shape, "col": col, "W": W}
        return Z

    def backward(self, grad):
        """
        Compute gradients w.r.t layer parameters and backward gradients.
        :param grad: gradients from previous layer
            with shape (batch_sz, out_h, out_w, out_c)
        :return d_in: gradients to next layers
            with shape (batch_sz, in_h, in_w, in_c)
        """
        # read size parameters
        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.ctx["X_shape"]
        pad_h, pad_w = self.padding[1:3]

        # grades w.r.t parameters
        flat_grad = grad.reshape((-1, out_c))
        d_W = self.ctx["col"].T @ flat_grad
        self.grads["w"] = d_W.reshape(self.kernel_shape)
        self.grads["b"] = np.sum(flat_grad, axis=0)



    def _inputs_preprocess(self, inputs):
        _, in_h, in_w, _ = inputs.shape
        k_h, k_w, _, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.padding_mode)

        return np.pad(inputs, pad_width=self.padding, mode="constant")
