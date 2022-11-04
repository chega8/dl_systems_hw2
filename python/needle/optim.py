"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        
        for param in self.params:
            self.u[param] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION    
        for param in self.params:
            if param.grad is None:
                continue
            if not param.requires_grad:
                continue
            
            u = self.u[param]
            grad = param.grad.data + param.data * self.weight_decay
            u = self.momentum * u + (1 - self.momentum) * grad
                
            param.data += (- self.lr) * u
            
            self.u[param] = u
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        
        for param in self.params:
            self.m[param] = 0
            self.v[param] = 0

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            if not param.requires_grad:
                continue
            
            m = self.m[param]
            v = self.v[param]
            
            grad = param.grad.data + param.data * self.weight_decay

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            param.data = param.data - self.lr * m_hat / (v_hat ** (1/ 2) + self.eps)
            
            self.m[param] = m
            self.v[param] = v
            
        ### END YOUR SOLUTION
