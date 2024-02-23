#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from typing import Callable
import math

# ### Activation function
# sigmoid, this takes in a single scalar and returns a scalar,
# use `np.vectorize` to apply this element wise
def sigmoid(v):
    return 1 / (1 + math.exp(-v))

# sigmoid derivative. acts and used the same as sigmoid above
def sigmoid_prime(z):
    sigmoid_z = sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z)

# definition of the mean square error function. subtract the target from
# the output, raise that to the power of 2 and find the mean of the vector.
def msqe(target, output):
    return np.average((target-output)**2)

# ## Network creation
#
# Our Network is a dataclass, which covers certain boilerplate things in our class
# It will store the weights, biases, activation function and activation derivative.
# purely so I don't need to rewrite all of this code for parts 4 and 5 (I hope)
#
#
# implementation does not usually use more than one activation function after
# all) then we take in a list of layers, where the first and last elements are the input and output layer respecively
#
# We then initalise our weight matrix and our bias vectors like so. for each
# layer we make a new matrix with the correct shape and initalise each element
# with a random number between $[0 \dots 1)$. We do the same thing for the bias
# vectors, dropping the first one as the input vector does not have biases.
@dataclass
class NeuralNetwork:
    theta: Callable
    theta_prime: Callable
    layers: list

    # We initalise the layers and count the number of layers
    def __post_init__(self):
        self.num_layers = len(self.layers)
        self.weights = [np.random.uniform(0, 1, size=(self.layers[i], self.layers[i + 1])) for i in range(self.num_layers - 1)]
        self.biases = [np.random.uniform(0, 1, size=(1, l)) for l in self.layers[1:]]

# ## Feed forward
#
# we take our input as x. `x` gets set as our first activation (which is also the
# last activation for the first run), we take out the last activation, do the
# dot product with the weight matrix and add the bias vector, saved in the name
# z.
#
# In other sources I have read this weighted sum is known as v, this has been
# a big source of confusion for me. In maths lingo this is known as
#
# $\vec{z} = \vec{x} \cdot \vec{w} + \vec{b}$ (1)
#
# We then take the $\vec{z}$, applying theta to each element in the resulting vector.
#
# $\vec{y} = \Theta(\vec{z})$ (2)
#
# $\vec{y}$ gets appeneded to the `activations` (ready for use in the next run)
# so does $\vec{z}$ into `zs` We save all of this information for back
# propagation.
#
# All of this book keeping may obscure whats actually happening. so the simpler
# implementation (my first implementation) is given below
    def feed_forward(self, x):
        biases = self.biases
        weights = self.weights
        theta = self.theta

        activations = [x]
        zs = []

        for b, w in zip(biases, weights):
            x = activations[-1]
            z = np.dot(x, w) + b
            activations.append(theta(z))
            zs.append(z)

        return activations, zs
# the simpler version as mentioned. We can see the equations pop out neatly.
    def feed_forward_result(self, x):
        biases = self.biases
        weights = self.weights
        theta = self.theta

        result = x

        for b, w in zip(biases, weights):
            z = np.dot(result, w) + b
            result = theta(z)

        return result

# ## Back Propagation
# Back propagation was (is) a pain in my back side to understand,
# but after reading it comes down to these 4 equations, and these 4 steps
# I realise that LaTeX notation is used here, that is because I had this all nicely
# typeset before I had to convert to python. Where its very much not clear I have converted it
# into a more conventional notation.
# the 4 equations are:
# 1. $\delta^L = \nabla_a C \odot \sigma'(z^L)$ **(1)**:
#    Error of the output layer
# 2. $\delta^l = ((w^{l+1})^T \dot \delta^{l+1}) \odot \sigma'(z^l)$ **(2)**:
#    Error of the current layer $l$, in terms of the next layer $l+1$
# 3. $\partial C / \partial b^l_j = \delta^l_j$ **(3)**:
#    Rate of change of the cost in respect to any bias in the network
# 4. $\partial C / \partial w^l_{jk} = a^{l-1}_k \dot \delta^l_j$ **(4)**:
#    Rate of change of the cost in respect to any weight of the network
#
# These equations look scary, because they are, but once you
# get past the notation and the index chasing of it, they become quite elegant,
# In the implementation I will highlight where each one pops out.
#
# The steps of the algorithm are as follows
# 1. Feed Forward input $x$, noting down $z^l = w^l \dot a^l + b^l$ and
# $a^l = \sigma'(z^l)$ where $l$ is each layer. My feed forward implementation already does this.
# 2. Calculate the output layer error $\delta^L$: $\delta^L = \nabla_a C \odot \sigma'(z^L)$ **(1)**
# 3. Backpropagate the error: Go backwards from the output layer, and calculate the error for the last layer.
#    In other words. For each $l = L - 1, L - 2, ..., 2$ calculate $\delta^l = ((w^{l+1})^T \dot \delta^{l+1}) \odot \sigma'(z^l)$ **(2)**
# 4. Output the gradient of the cost function (so we can decend that gradient).
#    $\partial C / \partial b^l_j = \delta^l_j$ **(3)**:
#    $\partial C / \partial w^l_{jk} = a^{l-1}_k \dot \delta^l_j$ **(4)**
    def back_propagate(self, input, target):
        # Initalise for later
        nabla_b = [None] * (self.num_layers - 1)
        nabla_w = [None] * (self.num_layers - 1)
        # Step 1, feed forward the input
        activations, zs = self.feed_forward(input)

        # Equation 1
        cost = activations[-1] - target
        delta_out = cost * self.theta_prime(zs[-1])
        delta_l = delta_out
        # Equation 3 for output layer
        nabla_b[-1] = delta_out
        # Equation 4 for output layer
        nabla_w[-1] = (activations[-2].T).dot(delta_l)
        # here we go from the second to last layer to the first layer
        # Python has this concept of negative indexing, where -1 can access the last element, -2 access the second to last and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            w = self.weights[-l+1]
            a = activations[-l-1]
            a = np.array(a).reshape(1, len(a))
            # Equation 2 for current layer l
            delta_l = delta_l.dot(w.T) * self.theta_prime(z)
            nabla_b[-l] = delta_l
            nabla_w[-l] = (a.T).dot(delta_l)

        return nabla_b, nabla_w

# ## Gradient Decent
# Now that we know  the directions we need to go, we need to decend it!
# Instead of working out the change all in one go, we will do it in batches
# We will work out how to decend one batch, get a new set of weights and biases, and then use that new network
# and do it again until we run out of batches!
# This is known as Stochastic gradent decent.
#
# Here we are decending one batch, we back propagate for each value of x and y,
# collecting the results. Once we have done that for all of the batches, we
# apply the learning rate to the current weights and biases.
    def decend_one_batch(self, eta, batch):
        delta_nabla_biases = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            nabla_b, nabla_w = self.back_propagate(x, y)
            delta_nabla_weights = [dnw + nw for dnw, nw in zip(delta_nabla_weights, nabla_w)]
            delta_nabla_biases  = [dnw + nw for dnw, nw in zip(delta_nabla_biases, nabla_b)]

        self.weights = [w - (eta * delta_w) for w, delta_w in zip(self.weights, delta_nabla_weights)]
        self.biases  = [b - (eta * delta_b) for b, delta_b in zip(self.biases, delta_nabla_biases)]


# In each epoch, we shuffle the data, partition it into groups of `batch_size`,
# and then loop over the batch decending once for each batch.
# Note there is no separate train function,
# performing gradient decent is our training
    def stochastic_gradient_decent(self, data, epochs, learning_rate, batch_size, test_data, test_result):
        eta = learning_rate / batch_size
        for epoch in range(epochs):
            shuffled = data.copy()
            np.random.shuffle(shuffled)
            batches = [shuffled[k:k+batch_size] for k in range(0, len(shuffled), batch_size)]
            for batch in batches:
                self.decend_one_batch(eta, batch)

            result = self.feed_forward_result(test_data)
            print(f"""{epoch = },
            {result = }
            msqe error = {msqe(test_result, result)}
            """)

# ## Training
# For the training data, we will be doing multiplication of 2 numbers

def generate_data(num_inputs):
    x1 = np.random.uniform(0, 1, size=num_inputs)
    x2 = np.random.uniform(0, 1, size=num_inputs)

    inputs = [np.array(e).reshape(1, 2) for e in list(zip(x1, x2))]
    targets = np.reshape(np.multiply(x1, x2), (num_inputs, 1))

    data = list(zip(inputs, targets))

    return data

# ## The final network
#
# Finally we can take our untrained network from before and forge it in fire,
if __name__ == '__main__':
    network = NeuralNetwork(np.vectorize(sigmoid), np.vectorize(sigmoid_prime), [2, 10, 1])
    untrained_output = network.feed_forward_result([0.2, 0.9])
    print(f"untrained {untrained_output=}")
    data = generate_data(100)
    network.stochastic_gradient_decent(data, 20, 0.12, 10, [0.2, 0.9], [0.2 * 0.9])
    print("done training")
    trained = network.feed_forward_result([0.2, 0.9])
    print(f"trained {trained =}")
    print(f"error = {msqe([0.2 * 0.9], trained)}")
