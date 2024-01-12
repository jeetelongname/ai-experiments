;; # Part 4 of the Coursework
;;
;; this part is considerably shorter and for the first time I am showing
;; what I import for this part!
;; mainly to show we can reuse all parts of part 3 to make part 4
(ns part4
  (:require part3
            [nextjournal.clerk :as clerk]
            [nextjournal.clerk.experimental :as cx]
            [clojure.core.matrix :as matrix]
            [clojure.core.matrix.random :as random]))

;; ## Activation Functions
;;
;; The rectified linear unit activation function. For everything below or equal
;; to zero, it is zero. otherwise its x. in otherwords, its the maximum of zero
;; and x.
(defn relu [z]
  (max 0 z))

;; the derivative is just 1 if its above zero,
;; zero if below zero undefined for zero.
(defn relu-derivative [z]
  (cond
    (> z 0) 1
    (< z 0) 0
    :else Double/NaN))

(def make-relu-network
  (partial part3/make-network
           relu
           relu-derivative))

(def ^:private
  untrained-network (make-relu-network 2 [5 5] 1))

(clerk/example
 (part3/feed-forward-result untrained-network [0.2 0.9]))

;; # knobs
^{::clerk/sync true
  ::clerk/viewer (partial cx/slider {:min 10 :max 1000})}
(def ^:private num-inputs (atom 100))

^{::clerk/sync true
  ::clerk/viewer (partial cx/slider {:min 1 :max 100})}
(def ^:private epochs
  (atom 50))

^{::clerk/sync true
  ::clerk/viewer (partial cx/slider {:min 0.0001 :max 0.5 :step 0.0001})}
(def ^:private learning-rate
  (atom 0.0064))

^{::clerk/sync true
  ::clerk/viewer (partial cx/slider {:min 1 :max @num-inputs})}
(def ^:private batch-size
  (atom 10))


(def data
  (let [num-inputs @num-inputs
        x1 (random/sample-uniform num-inputs)
        x2 (random/sample-uniform num-inputs)
        inputs (part3/zip x1 x2)
        targets (matrix/reshape
                 (matrix/mul x1 x2)
                 [num-inputs 1])]
    (part3/zip inputs targets)))

(def ^:private trained-network
  (part3/stochastic-gradient-decent
   untrained-network data
   :epochs @epochs
   :learning-rate @learning-rate
   :batch-size @batch-size
   :test-data [0.2 0.9]
   :test-result (* 0.2 0.9)))

^{::clerk/visibility {:code :fold}}
(clerk/plotly
 (let [epochs (trained-network :epochs)]
   {:data [{:x (mapv :epoch epochs)
            :y (mapv :msqe  epochs)
            :mode "lines"}]}))

^{::clerk/visibility {:code :fold}}
(let [test [0.2 0.9]
      target [(apply * test)]
      output (part3/feed-forward-result (trained-network :network) test)]
  (clerk/table
   [{"untrained" (part3/feed-forward-result untrained-network test)
     "trained"   output
     "target"    target
     "msqe"      (part3/mean-square-error target output)}]))

;; Done
