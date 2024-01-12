(ns index
  {:nextjournal.clerk/visibility {:code :hide :result :hide}}
  (:require [nextjournal.clerk :as clerk]))

(def notebooks
  [{:title "Genetic Algorithms using the one max problem"
    :introduction "The first part is about genetic algorithms using the very simple one max solution"
    :link "notebooks/part1.clj"}
   {:title "A Single Perceptron and linear seperability"
    :introduction "This part is about looking at a single neuron, "
    :link "notebooks/part2.clj"}
   {:title "A foray into Neural Networks and Back Propagation"
    :introduction "Here we graduate from one perceptron to many, looking into how now many perceptrons learn through back propagation"
    :link "notebooks/part3.clj"}
   {:title "Replacing Sigmoid with ReLU"
    :introduction "Here I basically reuse part three but play with the Rectified Linear Unit."}
   {:title "Back-Propagation, a high level overview"
    :introduction "Back propagation was just too interesting to leave on its own, so I have taken it apart, piece by piece, to then build it back up in terms we can understand"
    :link "notebooks/backpropagation.md"}])

(def notebook-count (count notebooks))

;; # AI experimentations.

;; This set of notebooks is my foray into Artificial intelligence and machine
;; learning. These are notes I produced for a class but they are not limited to
;; that class. In here there is a mix of Genetic Algorithms, A lot of Neural
;; Networks and maybe in the future a little image recognition. The plan is to keep
;; doing what I can using clerk and clojure and because a lot of this is just
;; learning who cares that its not performant or something.

{::clerk/visibility {:result :show}}

;; ## Notebooks

(clerk/html
 (into [:div]
       (map-indexed (fn [part {:keys [title introduction link]}]
              [:div
               [:a {:href (clerk/doc-url link)} [:strong title]]
               [:p introduction
                [:br]
                "This is Part " (inc part) " of " notebook-count]
               [:hr]]))
       notebooks))
