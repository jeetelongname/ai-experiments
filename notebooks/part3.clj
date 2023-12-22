;; # Part 3 of course work
^{:nextjournal.clerk/visibility :hide-ns}
(ns part3
  {:nextjournal.clerk/toc :collapsed}
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.experimental :as cx]
            [clojure.core.matrix :as matrix]
            [clojure.math :as math]))


(defn sigmoid [v]
  (/ 1.0 (+ (math/exp v)
            1)))
