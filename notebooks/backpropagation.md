# Back Propation, A high level overview
```clojure
(ns backpropagation
  {:nextjournal.clerk/visibility {:code :hide}}
  (:require [nextjournal.clerk :as clerk]))
```

# Introduction

The back propagation algorithm is one that is steeped in massive amounts of
maths and understanding, its not something that can be picked up simply since
its so rich. that being said its also not unotanably difficult to understand
Once you get past the notation, the index chasing, the sheer madness of it, it
becomes somwhat simple to grok, even if you have to treat parts as black boxes
that just kinda work. 

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

