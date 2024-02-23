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

# ## Activation function
# Unit step is an activation function
# that is 1 for non negative values
# 0 for negative values
# this can classify problems above a linear line, or below it
def unit_step(v):
    return 1 if v >= 0 else 0

# ## Perceptron
# - $\Theta$: activation function
# - x: input vector
# - w: weight vector
# - b: bias scalar
#
# $y = \Theta(\vec{w} \cdot \vec{x} + b)$
def perceptron(theta, x, w, b):
    result = np.dot(w, x) + b
    return theta(result)

def or_perceptron(x):
    return perceptron(
        unit_step,
        x,
        np.array([1, 1]),
        -1)

print("testing this or perceptron we get these outputs")
for k in or_data.keys():
    print(f"""input = {k},
    perceptron answer = {or_perceptron(k)}
    expected output = {or_data[k]}
    """)
print("perceptron can compute or!")
print("""this is to be expected as or is a linear seperable problem
But XOR is not, we can show this graphically here. Press q after the first graph""")
# # Plot 1
plt.scatter([0, 1, 1], [1, 0, 1], label='One', s=[10, 10, 10])
plt.scatter([0], [0], label='Zero', s=[10])
plt.plot([0, 0.9], [0.9, 0], label='Linear separator')

plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Output of the Or function over 2 inputs')
plt.legend()
plt.grid(True)

plt.show()
print("here we can look at XOR, there is no linear line we can draw, quadratic ones yes but linear no.")
# # Plot 2
plt.scatter([0, 1], [1, 0], label='One', s=[10, 10])
plt.scatter([0, 1], [0, 1], label='Zero', s=[10])

plt.xticks(range(2))
plt.yticks(range(2))
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Output of the XOR function over 2 inputs')
plt.legend()
plt.grid(True)

plt.show()

print("""quick look at the learning rule
i did not have time to properly pick up the learning rule of a single perceptron
but in theory it can be implemented like so.
""")

# the perceptron is an object that contains the weights,
# d is the position the weights should be in,
# either its above or below, 1 or -1
# the points is the points that need updating, we go through the ds and the points
# updating all the perceptron weights each time. according to the learning rate.
def update_weights(perceptron, ds, learning_rate, points):
    for d, x in zip(ds, points):
        perceptron.w = perceptron.w + learning_rate * d * x
