"""
Sample code automatically generated on 2022-05-14 15:35:36

by www.matrixcalculus.org

from input

d/db1 tanh(W1 * t + b1) = diag(vector(1)-tanh(b1+t*W1).^2)

where

W1 is a vector
b1 is a vector
t is a scalar

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(W1, b1, t):
    assert isinstance(W1, np.ndarray)
    dim = W1.shape
    assert len(dim) == 1
    W1_rows = dim[0]
    assert isinstance(b1, np.ndarray)
    dim = b1.shape
    assert len(dim) == 1
    b1_rows = dim[0]
    if isinstance(t, np.ndarray):
        dim = t.shape
        assert dim == (1, )
    assert W1_rows == b1_rows

    t_0 = np.tanh((b1 + (t * W1)))
    functionValue = t_0
    gradient = np.diag((np.ones(W1_rows) - (t_0 ** 2)))

    return functionValue, gradient

def checkGradient(W1, b1, t, h):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(h)
    f1, _ = fAndG(W1, b1 + t * delta, t)
    f2, _ = fAndG(W1, b1 - t * delta, t)
    f, g = fAndG(W1, b1, t)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData(h):
    W1 = np.random.randn(h)
    b1 = np.random.randn(h)
    t = np.random.randn(1)

    return W1, b1, t

if __name__ == '__main__':

    np.random.seed(23)
    h = 32
    W1, b1, t = generateRandomData(h)
    functionValue, gradient = fAndG(W1, b1, t)
    print('functionValue = ', functionValue)
    print('gradient is {0}:\n{1}'.format(gradient.shape, gradient))

    print('numerical gradient checking ...')
    checkGradient(W1, b1, t, h)
