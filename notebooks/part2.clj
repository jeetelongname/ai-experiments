;; # Part 2 of course work
^{:nextjournal.clerk/visibility :hide-ns}
(ns part2
  {:nextjournal.clerk/toc :collapsed}
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.experimental :as cx]
            [clojure.core.matrix :as matrix]
            [clojure.math :as math]))

;; ## Training data
;; Our data is the truth table of the function we are using,
;; our key is the input and our value is the output
(def or-data {[0 0] 0
              [0 1] 1
              [1 0] 1
              [1 1] 1})

(def xor-data {[0 0] 0
               [0 1] 1
               [1 0] 1
               [1 1] 0})

;; ## Activation function
;; Unit step is an activation function
;; that is 1 for non negative values
;; 0 for negative values
(defn unit-step [v]
  (if (>= v 0)
    1
    0))

;; ## Perceptron
;; $y = \Theta(w_{1}x_{1} + w_{2}x_{2} + b)$
;;
;; - $\Theta$: activation function
;; - x: input vector
;; - w: weight vector
;; - b: bias scalar
;;
;; $out = \Theta(\vec{w} \cdot \vec{x} + b)$
;; argument
(defn perceptron [ϴ x w b]
  (ϴ (+ (matrix/dot w x)
        b)))

;; ## Question 1, Solving the OR problem
;; For the OR function it will take in two inputs, with a weight vector
;; with two values. what we need to do is loop over and check that we get
;; the correct value no matter what.
;;
;; We need to think about the weights and the bias we will need to apply.
;; Essentially our or function turns on when the sum of the input vector is more
;; than or equal to 1. in that case we can out put equal weighting on both of
;; our inputs and set the bias to -1, this means for `[0 0]` our output before
;; applying $\Theta$ would be `-1` and for anything else it would
;; be more than or equal to zero. This is exactly what the or function does!
;;
;; In the case of code we can then define the or function as such

(defn or-perceptron [x]
  (perceptron unit-step x [1 1] -1 ))

;; we can then test this on all of the input.
^{::clerk/visibility {:code :fold}}
(clerk/table {:head ["Input" "Perceptron answer" "Test data"]
              :rows (mapv
                     ;; juxt "explodes a value" meaning it will apply the
                     ;; input value to each function. this allows us to put it all
                     ;; into a nice table.
                     (juxt identity
                           or-perceptron
                           or-data)
                     (keys or-data))})

;; ## Question 2, Not Solving the XOR problem

^{::clerk/visibility {:code :hide}}
(clerk/html
 [:a {:href (clerk/doc-url "notebooks/part3.clj")}
  "Part 3: Backpropagation"])
