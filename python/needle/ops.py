"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_input = node.inputs[0]
        return (out_grad * self.scalar * (node_input ** (self.scalar - 1)), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        l_input, r_input = node.inputs
        l_grad = Tensor(out_grad * divide(Tensor([1], dtype='float32'), r_input))
        r_grad = Tensor(out_grad * divide(-l_input, power_scalar(r_input, 2)))
        return l_grad, r_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar).astype(a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (divide(out_grad, Tensor([self.scalar], dtype='float32')), )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ax1, ax2 = len(a.shape) - 1, len(a.shape) - 2
        if self.axes is not None:
            ax1, ax2 = self.axes
            
        return array_api.swapaxes(a, ax1, ax2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ax1, ax2 = len(node.inputs[0].shape) - 1, len(node.inputs[0].shape) - 2
        if self.axes is not None:
            ax1, ax2 = self.axes
        return (out_grad.transpose(axes=(ax1, ax2)), )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return (out_grad.reshape(node.inputs[0].shape), )
        return (reshape(out_grad, node.inputs[0].shape), )
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_input_shape = node.inputs[0].shape
        axes = get_broadcast_backward_axes(node_input_shape, self.shape)
        return (out_grad.sum(tuple(axes)).reshape(node_input_shape), )
        ### END YOUR SOLUTION

def get_broadcast_backward_axes(input_shape, out_shape):
    axes_shape = []
    for i in range(len(out_shape)):
        if i + 1 > len(input_shape):
            axes_shape.append(i)
            continue

        if input_shape[i] != out_shape[i]:
            axes_shape.append(i)
    return axes_shape

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_input = node.inputs[0]
        new_shape = list(node_input.shape)
        
        if self.axes is None:
            new_shape = [1 for _ in new_shape]
        else:
            for i in self.axes:
                new_shape[i] = 1

        return out_grad.reshape(new_shape).broadcast_to(node_input.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        l_input, r_input = node.inputs
        l_grad, r_grad = out_grad @ r_input.transpose(), l_input.transpose() @ out_grad

        r_axes = get_axes_to_sum_mul_backward(r_input.shape, r_grad.shape)
        l_axes = get_axes_to_sum_mul_backward(l_input.shape, l_grad.shape)

        r_grad = r_grad.sum(axes=r_axes)
        l_grad = l_grad.sum(axes=l_axes)
        return l_grad, r_grad
        ### END YOUR SOLUTION


def get_axes_to_sum_mul_backward(input_shape, grad_shape):
    return tuple(range(len(grad_shape) - len(input_shape)))


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_input = node.inputs[0]
        grad = Tensor([1], dtype='float32') / node_input
        return (out_grad * grad, )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_input = node.inputs[0]
        return (out_grad * exp(node_input), )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0.)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        arr = node.realize_cached_data()
        arr = array_api.greater(arr, 0.).astype(array_api.float32)
        return out_grad * Tensor(arr, dtype=arr.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        self.cached_exp = None
        self.cached_sum = None

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            max_z = array_api.max(Z)
        else:
            max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        
        exp_ = array_api.exp(Z - max_z)
        self.cached_exp = exp_
        if self.axes is None:
            sum_ = array_api.sum(exp_)
        else:
            sum_ = array_api.sum(exp_, axis=self.axes, keepdims=True)
        
        self.cached_sum = sum_
        log_ = array_api.log(sum_)
        output = (log_ + max_z).squeeze()
        if len(output.shape) == 0 and self.axes is not None:
            output = array_api.expand_dims(output, axis=-1)
        return output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_input = node.inputs[0]
        tmp_shape = list(node_input.shape)
        if self.axes:
            for ax in self.axes:
                tmp_shape[ax] = 1
        else:
            tmp_shape = [1 for _ in tmp_shape]

        grad = reshape(out_grad, tmp_shape)
        grad = broadcast_to(grad, node_input.shape)

        softmax_out = Tensor(self.cached_exp / self.cached_sum)
        return (grad * softmax_out, )
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Maximum(TensorOp):
    def __init__(self, axes):
        self.axes = axes

    def compute(self, a: NDArray):
        return array_api.max(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        raise NotImplementedError()

def maximum(a, axes=None):
    return Maximum(axes=axes)(a)


def softmax(x):
    z = numpy.exp(x)
    return z / numpy.sum(z, axis=1, keepdims=True)


def softmax_tensor(x):
    z = exp(x)
    shape = z.shape
    return z / summation(z)