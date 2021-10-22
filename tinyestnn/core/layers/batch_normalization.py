from tinyestnn.core.initializer import Ones, Zeros
from tinyestnn.core.layers.layer import Layer


class BatchNormalization(Layer):
    def __init__(self,
                 momentum=0.99,
                 gamma_init=Ones(),
                 beta_init=Zeros(),
                 epsilon=1e-5):
        super().__init__()
        self.m = momentum
        self.epsilon = epsilon

        self.initializers = {"gamma": gamma_init, "beta": beta_init}
        self.reduce = None

        self.ctx = None

    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass
