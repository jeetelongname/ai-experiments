#!/usr/bin/env python3
#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from typing import Callable
import math

# ### Activation function

# The rectified linear unit activation function. For everything below or equal
# to zero, it is zero. otherwise its x. in otherwords, its the maximum of zero
# and x.
def relu(x):
    return max(0, x)

# the derivative is just 1 if its above zero,
# zero if below zero undefined for zero.
def relu_prime(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return np.nan
def msqe(target, output):
    return np.average((target-output)**2)

# ## Network creation
# Our Network is the same from part 3, I put the definition here for brevity.
# all of it is copied over sans the comments
@dataclass
class NeuralNetwork:
    theta: Callable
    theta_prime: Callable
    layers: list

    def __post_init__(self):
        self.num_layers = len(self.layers)
        self.weights = [np.random.uniform(0, 1, size=(self.layers[i], self.layers[i + 1])) for i in range(self.num_layers - 1)]
        self.biases = [np.random.uniform(0, 1, size=(1, l)) for l in self.layers[1:]]

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

    def feed_forward_result(self, x):
        biases = self.biases
        weights = self.weights
        theta = self.theta

        result = x
        for b, w in zip(biases, weights):
            z = np.dot(result, w) + b
            result = theta(z)

        return result

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
            # Equation 2 for current layer l
            delta_l = delta_l.dot(w.T) * self.theta_prime(z)
            nabla_b[-l] = delta_l
            nabla_w[-l] = (a.T).dot(delta_l)

        return nabla_b, nabla_w

    def decend_one_batch(self, eta, batch):
        delta_nabla_biases = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            nabla_b, nabla_w = self.back_propagate(x, y)
            delta_nabla_weights = [dnw + nw for dnw, nw in zip(delta_nabla_weights, nabla_w)]
            delta_nabla_biases  = [dnw + nw for dnw, nw in zip(delta_nabla_biases, nabla_b)]

        self.weights = [w - (eta * delta_w) for w, delta_w in zip(self.weights, delta_nabla_weights)]
        self.biases  = [b - (eta * delta_b) for b, delta_b in zip(self.biases, delta_nabla_biases)]

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

def generate_data(num_inputs):
    x1 = np.random.uniform(0, 1, size=num_inputs)
    x2 = np.random.uniform(0, 1, size=num_inputs)

    inputs = [np.array(e).reshape(1, 2) for e in list(zip(x1, x2))]
    targets = np.reshape(np.multiply(x1, x2), (num_inputs, 1))

    data = list(zip(inputs, targets))

    return data

# ## The final network
#
# again like before we can take our random network and train it, I have changed the learning rate to something that works better with ReLU
if __name__ == '__main__':
    network = NeuralNetwork(np.vectorize(relu), np.vectorize(relu_prime), [2, 5, 5, 1])
    untrained_output = network.feed_forward_result([0.2, 0.9])
    print(f"{untrained_output=}")
    data = generate_data(100)
    network.stochastic_gradient_decent(data, 20, 0.005, 10, [0.2, 0.9], [0.2 * 0.9])
    print("done training")
    trained = network.feed_forward_result([0.2, 0.9])
    print(f"{trained =}")
    print(f"error = {msqe([0.2 * 0.9], trained)}")
