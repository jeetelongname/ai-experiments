#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

or_data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 1
}

xor_data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

def unit_step(v):
    return 1 if v >= 0 else 0

def perceptron(theta, x, w, b):
    result = np.dot(w, x) + b
    return theta(result)

def or_perceptron(x):
    return perceptron(
        unit_step,
        x,
        np.array([1, 1]),
        -1
    )

# Plotting markers and lines
plt.scatter([0, 1, 1], [1, 0, 1], label='One', s=[10, 10, 10])
plt.scatter([0], [0], label='Zero', s=[10])
plt.plot([0, 0.9], [0.9, 0], label='Linear separator')

# Setting layout options
plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Output of the Or function over 2 inputs')
plt.legend()
plt.grid(True)

plt.show()

plt.scatter([0, 1], [1, 0], label='One', s=[10, 10])
plt.scatter([0, 1], [0, 1], label='Zero', s=[10, 10])

plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Output of the XOR function over 2 inputs')
plt.legend()
plt.grid(True)

plt.show()
