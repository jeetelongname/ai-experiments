;; # Part 3 of course work
^{:nextjournal.clerk/visibility :hide-ns}
(ns part3
  {:nextjournal.clerk/toc :collapsed}
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.experimental :as cx]
            [clojure.core.matrix :as matrix]
            [clojure.math :as math]))

;; ### Helpers
(def zip (partial map vector))

^{::clerk/visibility {:code :hide :result :hide}}
(fn [l l-len]
  {:der (mapv #(matrix/zero-matrix ;;
                (get l %)
                (get l (inc %)))
              (range (dec l-len)))
   :out (mapv #(matrix/zero-vector (get l %))
              (range l-len))})

;; ## Network creation
;;
;; Our Network is a map (dictionary in other languages)
;; It will store the weights, biases, activation function and activation derivative.
;; purely so I don't need to rewrite all of this code for parts 4 and 5 (I hope)
;;
;; we take in the functions first so I can create helpers later (one network
;; implementation does not usually use more than one activation function after
;; all) then we take in the size of our input vector, the hidden layers as a
;; vector containing the amount of neurons in each layer. and the size of the output vector.
;; eg, `2 [5 5] 1`
;;
;; We then initalise our weight matrix and our bias vectors like so. for each
;; layer we make a new matrix with the correct shape and initalise each element
;; with a random number between zero and one. We do the same thing for the bias vectors,
;; dropping the first one as the input vector does not have biases.
(defn make-network [activation-f activation-der
                    x-size hidden-layers y-size]
  (let [l (vec (flatten [x-size hidden-layers y-size]))
        l-len (count l)]
    {:weights (mapv #(matrix/emap
                      (fn [_] (rand))
                      (matrix/zero-array
                       [(get l %)
                        (get l (inc %))]))
                    (range (dec l-len)))
     :biases (into [] (comp
                       (map #(matrix/zero-vector (get l %)))
                       (drop 1)
                       (map (partial matrix/emap (fn [_] (rand)))))
                   (range l-len))
     :theta activation-f
     :theta-der activation-der
     :l l
     :l-len l-len}))

;; ## activation function
;; sigmoid, this takes in a single scalar and returns a scalar,
;; use `matrix/emap` to apply this element wise
(defn sigmoid [v]
  (/ 1.0 (+ (math/exp (- v))
            1)))

;; sigmoid derivative. acts and used the same as sigmoid above
(defn sigmoid-prime [x]
  (* x (- 1.0 x)))

;; helper mentioned before, I can now make a sigmoid network like so.
(def make-sigmoid-network (partial make-network
                                   sigmoid
                                   sigmoid-prime))

(def network (make-sigmoid-network 2 [5 5] 1))

;; ## Feed forward
;;
;; Our Feed Forward implementation takes in a network map and destructures it,
;; we take our input as x. `x` gets set as our first activation (which is also the
;; last activation for the first run), we take out the last activation, do the
;; dot product with the weight matrix and add the bias vector, saved in the name
;; z.
;;
;; In other sources I have read this weighted sum is known as v, this has been
;; a big source of confusion for me. In maths lingo this is known as
;;
;; $\vec{z} = \vec{x} \cdot \vec{w} + \vec{b}$ (1)
;;
;; We then take the $\vec{z}$, applying theta to each element in the resulting vector.
;;
;; $\vec{y} = \Theta(\vec{z})$ (2)
;;
;; $\vec{y}$ gets appeneded to the `activations` (ready for use in the next run)
;; so does $\vec{z}$ into `zs` We save all of this information for back
;; propagation.
;;
;; All of this book keeping may obscure whats actually happening. so the simpler
;; implementation (my first implementation) is given below
(defn feed-forward [{:keys [weights biases theta]} x]
  (reduce (fn [{activations :activations zs :zs} [b w]]
            (let [x (last activations)
                  z (matrix/add (matrix/dot x w) b)] ;; (1)
              ;; element map, apply this function to each element in the matrix (2)
              {:activations (conj activations (matrix/emap theta z))
               ;; save the weighted sum vector v for back propagation
               :zs (conj zs z)}))
          {:activations [x] :zs []} ;; Inital value
          (zip biases weights)))

(clerk/table
 (feed-forward network [0.2 0.1]))

(defn- feed-forward-simple [{:keys [weights biases theta]} x]
  (reduce (fn [x [b w]]
            (let [z (matrix/add (matrix/dot x w) b)]
              (matrix/emap theta z)))
          x (zip biases weights)))

(feed-forward-simple network [0.2 0.1])
;; Showing we get the same final answer

;; ## Back Propagation
;; Back propagation was (is) a pain in my back side to understand,
;; but after reading it comes down to these 4 equations, and these 4 steps
;;
;;
;; the 4 equations are:
;; 1. $\delta^L = \nabla_a C \odot \sigma'(z^L)$ **(1)**:
;;    Error of the output layer
;; 2. $\delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma'(z^l)$ **(2)**:
;;    Error of the current layer $l$, in terms of the next layer $l+1$
;; 3. $\frac{\partial C}{\partial b^l_j} = \delta^l_j$ **(3)**:
;;    Rate of change of the cost in respect to any bias in the network
;; 4. $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \cdot \delta^l_j$ **(4)**:
;;    Rate of change of the cost in respect to any weight of the network
;;
;; I mostly wanted to include them because they look scary, and they also look
;; nice all typeset. These equations look scary, because they are, but once you
;; get past the notation and the index chasing of it, they become quite elegant,
;; That being said, there is no royal road to understanding and you will need to
;; do some of the reading for yourself [^1]. Also note, $\odot$ means the
;; Hadamard product or element wise multiplication in non maths talk.
;;
;; The steps of the algorithm are as follows
;; 1. Feed Forward input $x$, noting down $z^l = w^l \cdot a^l + b^l$ and
;; $a^l = \sigma'(z^l)$ where $l$ is each layer. My feed forward implementation already does this.
;; 2. Calculate the output layer error $\delta^L$: $\delta^L = \nabla_a C \odot \sigma'(z^L)$ **(1)**
;; 3. Backpropagate the error: Go backwards from the output layer, and calculate the error for the last layer.
;;    In other words. For each $l = L - 1, L - 2, ..., 2$ calculate $\delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma'(z^l)$ **(2)**
;; 4. Output the gradient of the cost function (so we can decend that gradient).
;;    $\frac{\partial C}{\partial b^l_j} = \delta^l_j$ **(3)**,
;;    $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \cdot \delta^l_j$ **(4)**
;;
;;
;; [^1]: For my comrades who are reading this in an effort to grok it for
;; yoursef. Ill provide resources, diagrams and worded versions of the equations
;; and code in a separate document. If you are reading this Ioannis, you can
;; check it out too, though I would assume you would get less out of it.
(defn back-propagate [{:keys [theta-der l l-len weights biases] :as network} x target]
  (let [{:keys [activations zs]} (feed-forward network x) ;; Step 1
        ;; go through everything backwards.
        zs (vec (reverse zs))
        activations (vec (reverse activations))
        weights (vec (reverse weights))
        ;; Eq (1), Step 2
        error (matrix/sub (first activations) target)
        delta-out (matrix/mul error (matrix/emap theta-der (first zs)))]
    (reduce (fn [{:keys [delta-l nabla-b nabla-w]} layer]
              (let [z (get zs layer)
                    w (get weights (dec layer))
                    a (get activations layer)
                    ;; Eq (2), Step 3
                    delta-l (matrix/mul (matrix/dot
                                         delta-l
                                         (matrix/transpose w))
                                        (matrix/emap theta-der z))]
                ;; step 4
                {:delta-l delta-l
                 ;; Eq (3)
                 :nabla-b (assoc nabla-b layer delta-l)
                 ;; Eq (4)
                 :nabla-w (assoc nabla-w (dec layer)
                                 (matrix/dot
                                  delta-l
                                  (matrix/transpose a)))}))
            {:delta-l delta-out
             :nabla-b (mapv (constantly nil) l)
             :nabla-w (mapv (constantly nil) (range (dec l-len)))}
            (vec (range 1 (dec l-len))))))

(clerk/example
 (let [x [0.2 0.1]
       y (apply * x)]
   (back-propagate network x y)))

;; I am using stochastic gradient decent
(defn gradient-decent [network learning-rate])

(defn train [network x target epochs learning-rate])
