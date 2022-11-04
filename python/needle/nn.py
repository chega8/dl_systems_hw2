"""The module.
"""
from array import array
from typing import List, Callable, Any
from needle.autograd import Tensor, cpu
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.b = bias

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype)
        self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape((1, out_features)), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m = X.shape[0]
        output = X @ self.weight
        if self.b:
            output = output + self.bias.broadcast_to((m, self.out_features))
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        m = X.shape[0]
        shapes = X.shape[1:]
        new_dim = 1
        for i in shapes:
            new_dim *= i
        return ops.reshape(X, (m, new_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m, n = logits.shape
        y_one_hot = np.zeros((y.shape[0], logits.shape[-1]))
        y_one_hot[np.arange(y.shape[0]), y.numpy()] = 1
        y_one_hot = Tensor(y_one_hot, dtype='float32')
        loss = ops.logsumexp(logits, axes=(1,)) - ops.summation(
            logits * y_one_hot, axes=(1,)
        )
        return ops.summation(loss) / m
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # self.weight = init.constant(dim, c=1.0, device=device, dtype=dtype)
        # self.weight = Parameter(self.weight.reshape((1, self.dim)))
        # self.bias = init.constant(dim, c=0.0, device=device, dtype=dtype)
        # self.bias = Parameter(self.bias.reshape((1, self.dim)))
        self.weight = Parameter(init.constant(dim, c=1.0, device=device, dtype=dtype))
        self.bias = Parameter(init.constant(dim, c=0.0, device=device, dtype=dtype))
        self.running_mean = init.constant(dim, c=0.0, device=device, dtype=dtype)
        self.running_var = init.constant(dim, c=1.0, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, n = x.shape
        
        shaped_weight = self.weight.reshape((1, self.dim)).broadcast_to((m, n))
        shaped_bias = self.bias.reshape((1, self.dim)).broadcast_to((m, n))
        
        if self.training:
            E = x.sum(axes=(0,)) / m
            var_arg = x - E.reshape((1, n)).broadcast_to((m, n))
            running_var = (var_arg ** 2).sum(axes=(0,)) / m
            var = ((running_var + self.eps) ** (1 / 2)).reshape((1, n)).broadcast_to((m, n))
        
            self.running_mean.data = self.update_stat(self.running_mean.data, E.data)
            self.running_var.data = self.update_stat(self.running_var.data, running_var.data)

            return shaped_weight * (var_arg / var) + shaped_bias
        
        else:
            running_mean_shaped = self.running_mean.reshape((1, self.dim)).broadcast_to((m, n))
            running_var_shaped = self.running_var.reshape((1, self.dim)).broadcast_to((m, n))
            
            num = x - running_mean_shaped.data
            den = (running_var_shaped.data + self.eps) ** (1 / 2)
            return shaped_weight.data * (num / den) + shaped_bias.data
        ### END YOUR SOLUTION

    def update_stat(self, old_stat, observed_stat) -> Tensor:
        return (1 - self.momentum) * old_stat + self.momentum * observed_stat


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.constant(dim, c=1.0, device=device, dtype=dtype))
        self.bias = Parameter(init.constant(dim, c=0.0, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, n = x.shape
        
        shaped_weight = self.weight.reshape((1, self.dim)).broadcast_to((m, n))
        shaped_bias = self.bias.reshape((1, self.dim)).broadcast_to((m, n))

        E = ops.summation(x, axes=(1,)) / n
        sigma_arg = x - ops.broadcast_to(ops.reshape(E, (m, 1)), (m, n))
        sigma = ops.summation(ops.power_scalar((sigma_arg), 2), axes=(1,)) / n
        sigma = ops.power_scalar(sigma + self.eps, 1 / 2)
        sigma = ops.broadcast_to(ops.reshape(sigma, (m, 1)), (m, n))

        return shaped_weight * (sigma_arg / sigma) + shaped_bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            probs = init.randb(*x.shape, p=1 - self.p, device=x.device)
            probs /= 1 - self.p
            return x * probs
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
