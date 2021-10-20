import numpy as np
from tinyestnn.core.initializer import XavierUniform, Zeros
from tinyestnn.core.layers.layer import Layer


class Dense(Layer):
    """A dense layer operates `outputs = dot(intputs, weight) + bias`
    :param num_out: A positive integer, number of output neurons
    :param w_init: Weight initializer
    :param b_init: Bias initializer
    """
    def __init__(self,
                 num_out,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.initializers = {'w': w_init, "b": b_init}
        self.shape = {"w": [None, num_out], "b": [num_out]}

        self.ctx = None

    def forward(self, inputs):
        if not self.is_init:
            self.shape["w"][0] = inputs.shape[1]
            self._init_params()

        self.ctx = {"inputs": inputs}

        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.ctx["inputs"].T @ grad
        self.grads["b"] = np.sum(grad, axis=0)

        return grad % self.params["w"].T

    @property
    def param_names(self):
        return "w", "b"
