# Back Propation, A high level overview
```clojure
(ns backpropagation
  {:nextjournal.clerk/visibility {:code :hide}
   :nextjournal.clerk/toc :collapsed}
  (:require [nextjournal.clerk :as clerk]))

^{::clerk/visibility {:result :hide}}
(def mermaid-viewer
  {:transform-fn clerk/mark-presented
   :render-fn '(fn [value]
                 (when value
                   [nextjournal.clerk.render/with-d3-require {:package ["mermaid@8.14/dist/mermaid.js"]}
                    (fn [mermaid]
                      [:div {:ref (fn [el] (when el
                                             (.render mermaid (str (gensym)) value #(set! (.-innerHTML el) %))))}])]))})
```

# Introduction

The back propagation algorithm is one that is steeped in massive amounts of
maths and understanding, its not something that can be picked up simply since
its so rich. that being said its also not unotanably difficult to understand
Once you get past the notation, the index chasing, the sheer madness of it, it
becomes somewhat simple to grok, even if you have to treat parts as black boxes
that just kinda work. 

I will assume that you know *something* about neural networks. If not, I suggest
reading over the resources linked below, praying or both.

# Resources

To start off I will share some resources I used to get started, I will be
honest, it took many readings, many watchings, one *near* mental break down and a
christmas break to fully get everything, This guide will probably not help you
get it the first time as well but I hope its a start.

- http://neuralnetworksanddeeplearning.com/chap2.html 
  This Chapter of Michal Nielsens book goes over the entire algorithim from the
  cost to each equation to concrete code, its dense but human readable and a
  very good resource

- [3 blue 1 browns video series on the matter](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  This playlist goes for a more intuitive understanding of whats going on, its
  helped me make confections between the concrete maths and what I will need to
  code.

If people have more resources do let me know but these are the two main. 
to supliment these two do also check out.
- [another video on the matter](https://www.youtube.com/watch?v=iyn2zdALii8) 
- [3b1b's series on linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&pp=iAQB)

# The 4 Equations
  What I will do is introduce these 4 equations, full syntax and all, *then*
  break down what each term means and where you can get them from. The breaking
  down and build back up strategy is what I look for when I want to take
  straight uncut maths into code.
  
 1. $\delta^L = \nabla_a C \odot \sigma'(z^L)$ **(1)**:
    Error of the output layer
 2. $\delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma'(z^l)$ **(2)**:
    Error of the current layer $l$, in terms of the next layer $l+1$
 3. $\frac{\partial C}{\partial b^l_j} = \delta^l_j$ **(3)**:
    Rate of change of the cost in respect to any bias in the network
 4. $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \cdot \delta^l_j$ **(4)**:
    Rate of change of the cost in respect to any weight of the network
    
## Prerequisites
Some of the notation is found througout these 4 equations. its worth getting
through them now
- All of the superscripting is **NOT** exponentiation. These are indexes that
  are telling you where information is coming from, unless its a `T` which is
  the transposition operator for whatever reason. subscripts are also indexes. 
  This is why I called the algorithm index chasing earlier.
- $\odot$ is element wise multiplication, This can be called the Hadamard product or the Schur product.
  I will call it element wise multiplcation, because thats what it is. An
  example.

```clojure
(clerk/tex "
\\begin{bmatrix}
 1 \\\\
 2
\\end{bmatrix} \\odot 
\\begin{bmatrix}
3 \\\\
4
\\end{bmatrix} =
\\begin{bmatrix}
 1 \\cdot 3 \\\\
 2 \\cdot 4
\\end{bmatrix} =

\\begin{bmatrix}
3 \\\\
8
\\end{bmatrix}
")
```
- $\frac{\partial C}{\partial x^l_{jk}}$ and anything that looks like it, *is
  the gradient of the cost function in relation to whatever $x$ is*. In other
  words you can treat it as one term, the answers we are looking for
- $\delta$ is the error. We will discuss what the different forms of the error
  as we go through the 4 equations
- $\sigma'$ is the derivative of the sigmoid function, this is something you can
  copy into your code.


## Equation 1: Error of the output layer 
Like this says, we are looking for the Error of the output layer. 

$\delta^L = \nabla_a C \odot \sigma'(z^L)$

- $L$: is just the number of layers, as I said before superscripts are indexes.
- $l$: is the layer we are on. think of it as the `i` in a for loop.
- $\delta^L$: this means that this term is the error at $L$ in other words the
  error of the output layer! thats all it is.
- $z^L$: $z$ is defined as $w \cdot a + b$ in other words, its the output of
  feed forward without applying the activation function! 
  $z^l = w^l \cdot a^l + b^l$ for any layer $l$, that means $z^L$ is $z$ for the
  output layer. Note you don't need to do anything new for this, as this is the
  same code we are using in feed forward all we need to do is save the output of
  $w^l \cdot a^l + b^l$ when doing feed forward. 
- $\nabla_a C$: is defined as the rate of change of the cost function. In other
  words its the derivative of the cost function, in other words all this scary
  term is, is $a^L - y$. Where $a^L$ is the output of the NN, and $y$ is the
  target vector. Thats it. 
  
  We can write this same equation out as such
  
  $\delta^L = (a^L - y) \odot \sigma'(z^L)$ 
  
Which is much more approachable. In words we could describe this equation as.

> The Error of the output layer is equal to, the Activation of the output
> layer vector subtracted from the target vector, element multiplied with,
> the derivative of sigmoid applied to the z quantity vector.

In python this code can be defined as
```python
delta_L = (activations[-1] - target) * sigma_der(zs[-1])
```
activations is a list of all of the activations of the network, same for zs for
the z quantity, -1 takes out the last element of a python list.

## Equation 2: Error of the current layer in terms of the next layer. 
This equation allows us to *back propage* the error to each layer. 
In this case the next layer. If we are currently on the layer L2, we would work
out the error in terms of the output layer, if we were in L1, we would work out
the error in terms of L2. so on.

```clojure
(clerk/with-viewer mermaid-viewer
  "graph TD;
      Input_Layer-->L1;
      L1-->L2;
      L2-->Output_Layer;")
```
Here is a graph to help visulise it.

Now here is the equation.
$\delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma'(z^l)$

- $\delta^l$: error of the current layer
- $(w^{l+1})^T$: the weight matrix of the next layer
  [transposed](https://en.wikipedia.org/wiki/Transpose) Otherwise known as
  fliping diagonally. [^1]
- $\delta^{l+1}$  The error of the next layer
- $z^l$ the z quantity of the current layer.

This equation needs to be calcualted for every layer, we will discuss how this
is done for each when we discuss the steps of the algorithm

In words this alogorithm is saying
> The Error of the current layer is equal to, the weights of the next layer
> transposed and dot producted with the error of the next layer. This term is then
> element multiplied with the sigmoid derivative applied to the current layers z
> quantity.

in python

```python
delta_l = np.dot(weights[l+1].transpose(), delta_l) * sigmoid_der(z[l])
```
`delta_l` in the right hand side is the last error vector. `l` is the current
layer. 

[^1]: Side note, Numpy kinda supports this superscript `T` syntax. You can
    use the `ndarry.T` accessor like so, its defined as an alias to
    `ndarry.transpose()`. You can read about it [here]( https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html)
## Equation 3: Rate of change of the cost in respect to any bias in the network
 $\frac{\partial C}{\partial b^l_j} = \delta^l_j$

 We are nearly there gamers. 
 
 We have some new notation, there is this new subscript value j. This is to
 signify what neuron we are talking about. 
 This equation is saying, the value we need to use to change the bias of neuron
 j in layer l, is equal to the error of the neuron j at layer l. 
 In reality we don't need to think about that and we can say
 ```python
 nabla_b[l] = delta_l
 ```
   where `nabla_b` is storing all of the changes to all of the biases. 
   and `delta_l` is what equation 2 calculated. and `l` is the layer we are on.

In Words this equation is saying
> The change of the baises in layer l is equal to, the error vector of the
> current layer l.
## Equation 4: Rate of change of the cost in respect to any weight of the network

 $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \cdot \delta^l_j$
 
$k$ in this equation is meant to signify the last layer, rememeber that weights
represent *connections* and because of that they need a start ($k$) and an end
($j$). That all being said there is an easier way to talk about this equation.

 $\frac{\partial C}{\partial w} = a_{in} \cdot \delta_{out}$
 
 for any weight $w$, this is much easier to reason about. IMHO.
 
in words this would be
> The change of the weights l is equal to, the activation in multiplied with the
> error out.

In Python. 
```python
nabla_w[l] = np.dot(activations[l-1].transpose(), delta_l)
```

The reason we do the transposition is because we are trying to apply a
transformation to the error vector, its vector matrix multiplication. If that does not
click for you then I would not worry about it. Check out [three blue one browns
video on the dot product.](https://www.youtube.com/watch?v=LyGKycYT2v0). 
Depending on the language you are using you may need to reshape the vectors of x
size into 1 by x matrix. you can do that like so: 

```python
matrix = vector.reshape(1, vector.shape[0])
```

You can then transpose this like normal.
This is the reason I could not crack the last equation for anyone who is
wondering.
# The 4 steps of the algorithm. 

The steps of the algorithm are as follows:

1. Feed Forward input $x$, noting down $z^l = w^l \cdot a^l + b^l$ and
$a^l = \sigma'(z^l)$ where $l$ is each layer. I will assume your feed forward
implementation already does this.
2. Calculate the output layer error $\delta^L$: $\delta^L = \nabla_a C \odot \sigma'(z^L)$
3. Backpropagate the error: Go backwards from the output layer, and calculate the error for the last layer.
   In other words. For each $l = L - 1, L - 2, \dots, 2$ calculate $\delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma'(z^l)$
4. Output the gradient of the cost function (so we can decend that gradient).
   $\frac{\partial C}{\partial b^l_j} = \delta^l_j$
   $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \cdot \delta^l_j$

The way we should think about the algorithm is the first two steps "seed"
the algorithm, the next two back propagate the error through the layers.

If i were to write this in a python like pseudocode. 

```python
def backprop(network, x, target):
  # initialise these for later.
  nabla_biases = [None for _ in network.biases]
  nabla_weights = [None for _ in network.weights]
  
  # Seed of the algorithm.
  activations, zs = feedforward(network, x)

  # Eq 1
  error = activations.last - target
  delta_L = error * sigmoid_derivative(zs.last)
  delta_l = delta_L

  # set the changes in the output layer
  # Eq 3
  nabla_biases.last = delta_l
  # Eq 4
  nabla_weights.last = dot(activations[-2].transpose(), delta_l)
  for layer in range(2, network.num_layers):
    # using pythons negative indexing we can go through everything backwards
    z = zs[-l]
    
    # Eq 2
    delta_l = dot(network.weights[-l+1].transpose(), delta_l) * sigmoid_prime(z)
    # Eq 3
    nabla_biases[-l] = delta_l 
    # Eq 4
    nabla_weights[-l] = dot(activations[-l+1].transpose(), delta_l) 
    
  return nabla_weights, nabla_biases 
```

Refer to the 4 equations, you should see them pop out, you should also be able
to see how everything connects together. Once you understand what each part is
doing its not a hard thing to implement. How it does it, is another question. If
you want to get into that check out the resources again. notably read Michael
Nielsen's [chapter on the
matter](http://neuralnetworksanddeeplearning.com/chap2.html). You can find an
actual python implementation near the end of the chapter (just replace `xrange`
with `range`). You can then take these changes and use them in gradient decent,
I won't discuss that here, you can check out the resources, Ioannis's
implementation. or my code found in [part 3](/notebooks/part3.clj) if you like
clojure.

# Conclusions 

We have covered the back propagation algorithm, breaking down each equation into
things we can reason about, framing it in a couple of ways to hopefully make it
easier to grasp at, after this decomposition and rebuilding we have gone through
the steps of the algorithm. 
Hopefully this demystifies the algorithm. along side the resources provided you
should not have problems handling this yourself! Good luck comrades 🫡.
