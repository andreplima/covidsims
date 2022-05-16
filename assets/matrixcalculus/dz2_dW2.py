"""
Sample code automatically generated on 2022-05-14 15:12:44

by www.matrixcalculus.org

from input

d/dW2 W2 * z1 + b2 = z1'\otimes eye

where

W2 is a matrix
b2 is a vector
z1 is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(W2, b2, z1):
    assert isinstance(W2, np.ndarray)
    dim = W2.shape
    assert len(dim) == 2
    W2_rows = dim[0]
    W2_cols = dim[1]
    assert isinstance(b2, np.ndarray)
    dim = b2.shape
    assert len(dim) == 1
    b2_rows = dim[0]
    assert isinstance(z1, np.ndarray)
    dim = z1.shape
    assert len(dim) == 1
    z1_rows = dim[0]
    assert W2_rows == b2_rows
    assert z1_rows == W2_cols

    functionValue = (b2 + (W2).dot(z1))
    gradient = np.einsum('ij, k', np.eye(W2_rows, W2_rows), z1)

    return functionValue, gradient

def checkGradient(W2, b2, z1, h):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, h)
    f1, _ = fAndG(W2 + t * delta, b2, z1)
    f2, _ = fAndG(W2 - t * delta, b2, z1)
    f, g = fAndG(W2, b2, z1)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData(h):
    W2 = np.random.randn(3, h)
    b2 = np.random.randn(3)
    z1 = np.random.randn(h)

    return W2, b2, z1

if __name__ == '__main__':
  
    np.random.seed(23)
    h = 32
    W2, b2, z1 = generateRandomData(h)
    functionValue, gradient = fAndG(W2, b2, z1)
    print('functionValue = ', functionValue)
    print('gradient is {0}:\n{1}'.format(gradient.shape, gradient))

    print('numerical gradient checking ...')
    checkGradient(W2, b2, z1, h)
