;; # Part 1 of course work
^{:nextjournal.clerk/visibility :hide-ns}
(ns part1
  {:nextjournal.clerk/toc :collapsed}
  (:require [nextjournal.clerk :as clerk]
            [nextjournal.clerk.experimental :as cx]))

;; ## Parameters that control the algorithm
;;
;;These are the knobs we can fiddle
;; with to get different results and tune the algorithm. You can use the
;; generation slider in particular to go through each generation and see it
;; improve. **NOTE:** the number in the code is the *default* not the actual
;; value, see the top of the slider for that.
^{::clerk/sync true ::clerk/viewer (partial cx/slider {:min 2 :max 100 :step 2})}
(defonce population-size (atom 10))

^{::clerk/sync true ::clerk/viewer (partial cx/slider {:min 0 :max 100})}
(defonce generations (atom 20))

^{::clerk/sync true ::clerk/viewer (partial cx/slider {:min 1 :max 32})}
(defonce mutation-rate (atom 6))

;; ## Question 1. Correct creation of the population
;;
;; here we need to generate an individual,
;; An individual is represented as a vector of ints,
;; rand-int will generate numbers from $[0 ... n)$
;; In this case our genome size is 32 so we take 32 random values like so.
;; We convert to a vector using `vec` for performance reasons
(defn generate-individual []
  (vec (take 32 (repeatedly #(rand-int 2)))))

^{::clerk/auto-expand-results? true}
(generate-individual)

;; we can then generate a population by repeatedly calling `generate-individual`
(def population (vec (repeatedly @population-size generate-individual)))

;; ## Question 2 Correct use of a fitness function
;;
;; In this case our fitness function is looking for the amount of 1s in an
;; individual, In this case to find them we can count up the amount of 1s and 0s
;; in each individual like so.
^{::clerk/visibility {:result :hide}}
(def example-fitness
  (frequencies (generate-individual)))

^{::clerk/visibility {:code :hide}}
(clerk/col
 (clerk/table [example-fitness]))

;; We wish to pressure our individuals into a form where we have 32 1s and 0 0s.
;; In that case we can say our score is the amount of ones. Otherwise known as
;; the sum of the individual
(defn fitness [individual]
  (apply + individual))

;; this is our score for a given individual
^{::clerk/visibility {:code :fold}}
(let [individual (generate-individual)]
  (clerk/table {:head ["Fitness" "Individual"]
                :rows [((juxt fitness identity) individual)]}))

;; to get the entire fitness of our population we map the fitness and sort decending
(->> population
     (map fitness)
     (sort >))

;; ## Question 3. Correct use of GA operations
;; ### selection
;;
;; selection is the process of selecting individuals for the next generation.
;; The form I will be using is a weighted random selection, otherwise known as
;; roullete wheel selection. Our select function will calculate the cumulative
;; fitness then will select $n$ amount of times where $n$ is the population size.
(defn select [population fitnesses]
  (let [total-fitness (apply + fitnesses)
        normalised (map #(/ %1 total-fitness) fitnesses)
        cummulative (reductions + normalised)]
    (mapv    ;; mapv returns a vector, otherwise its is `map`
     (fn [_] ;; ignore the actual value, we don't need it
       (loop [rand-n (rand)
              ;; iterate through the population and cumulative sum at the same time
              [curr-cum & rest-cum] cummulative
              [curr-ind & rest-ind] population]
         (if (<= rand-n curr-cum)
           ;; we return the current individual,
           ;; which becomes part of the new population
           curr-ind
           ;; recur enters the next loop, we take off the current values
           (recur rand-n rest-cum rest-ind))))
     (range @population-size))))

^{::clerk/auto-expand-results? true
  ::clerk/budget nil
  ::clerk/visibility {:code :fold}}
(clerk/table
 {:head ["Current" "Selected"]
  :rows (mapv vector
              population
              (select population (map fitness population)))})

;; ### mutation
;;
;; Mutation is when we select random alliel and twiddle it.

;; this helper gets $n$ random indexes, we use a set to make sure each is unique.
;; We start off by taking the amount amount of mutation values,
;; we put them into a set to make sure each is unique
;; then if the amount of indexes is less than the number needed we add one more
;; to the set and check again.
(defn get-indexes [n]
  (let [allele-generator #(rand-int 32)]
    (loop [index-set (set (take n (repeatedly allele-generator)))]
      (if (= (count index-set) n)
        index-set
        (recur (conj index-set (allele-generator)))))))

(defn mutate [mutation-value individual]
  (let [indexes (get-indexes mutation-value)]
    ;; go through the list of [index new-val] and set each index to new-val
    ;; we use reduce here because its the equiv of a stateful loop
    (reduce (fn [acc [index to-set]]
              (assoc-in acc [index] to-set)) individual
            (map vector indexes (repeat mutation-value 1)))))

(mutate @mutation-rate (vec (take 32 (repeat :example))))
;; ### crossover
;;
;; crossover is when we take two parents and cut them at some kind of random
;; interval, this gives us two new children.
;;
;; this helper function takes a vec and splits it, returning the two halfs,
;; this is an $O(1)$ operation which is the main reason we used vectors
;; in the first place
(defn vec-split [vec cut-point]
  [(subvec vec 0 cut-point)
   (subvec vec cut-point)])

(defn crossover
  [[ind1 ind2]]
  (let [cut-point (rand-int 33)
        [ind1-f ind1-l] (vec-split ind1 cut-point)
        [ind2-f ind2-l] (vec-split ind2 cut-point)]
    [(vec (concat ind1-f ind2-l)) (vec (concat ind2-f ind1-l))]))

(clerk/table
 (crossover [(vec (repeat 32 0))
            (vec (repeat 32 1))]))

;; ## Question 4. Demonstrably working GA

;; ### Termination
;; our termination conditions is when we either get an individual with 32 1s or we will reach `@generations`
(defn terminate? [fitness generation]
  (or (= (->> fitness
              (sort >)
              first)
         32)
      (= generation @generations)))

;; This helper does one pass over the population, performing crossover and
;; mutation on all of the individuals in the population.
(defn evolve [population]
  (into []
        (comp
         ;; group individuals into two parents
         (partition-all 2)
         ;; apply crossover to these parents
         (map crossover)
         ;; flatten the list
         cat
         ;; apply mutation to each individual
         (map (partial mutate @mutation-rate)))
        population))

;; ### Training
;; the training function brings this all together, we loop u
(defn train [population]
  (loop [pop population
         generation 0]
    (let [fitnesses (mapv fitness pop)]
      (if (terminate? fitnesses generation)
        pop ;; return the final population
        ;; If we are not happy we take the population,
        ;; select it, then evolve it.
        ;; recur starts the loop again
        (recur (-> pop
                   (select fitnesses)
                   evolve)
               (inc generation))))))

;; this gives us our final look, at the data, we can compare the trained population compared to its original,
;; I have sorted it by fitness but the changes should be evident.
^{::clerk/visibility {:code :fold}}
(clerk/table
 {:head ["Individual" "Fitness" "Original" "Fitness"]
  :rows
  (mapv (partial apply conj)
        (mapv (juxt identity fitness)
              (sort-by fitness > (train population)))
        (mapv (juxt identity fitness)
              (sort-by fitness > population)))})

;; finished
^{::clerk/visibility {:code :hide}}
(clerk/html [:a {:href (clerk/doc-url "notebooks/part2.clj")} "Part 2: Perceptron and the XOR problem"])
