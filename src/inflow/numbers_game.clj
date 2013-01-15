;; This is the number game, from Section 3.2 of "Machine Learning, a
;; Probabilistic Perspective" by Kevin Murphy.  It is given as an
;; example of using inflow.
;;
;; The game works like this.  You shown a set of numbers and have to
;; guess the 'natural set' from which they arise: the even numbers, odd
;; numbers, powers of 2, etc.  This shows how to play this game like a
;; Bayesian Boss.
;;
(ns inflow.numbers-game
  (:use [incanter.charts :only [bar-chart]]
        [incanter.core   :only [pow view]]
        [clojure.core.incubator :only [dissoc-in]]
        [clojure.set     :only [rename-keys union difference]])
  (:require
   [inflow.core :as inf]))

;; -----------------------------------------------------------------------------
;; Some helpful utilities

;; Arbitrarily set 100 as the maximum number.
(def max-num 100)

(defn <=Max [xs]
  (->> xs
       (take-while #(<= % max-num))
       (map double)))

(defn to-hyp [xs]
  (apply sorted-set (<=Max xs)))

;; -----------------------------------------------------------------------------
;; Our collection of hypotheses.
;; They work simply by giving the set of all members of the hypothesis (its extension).
;; These are just (sorted) sets, so inclusion of
;; data in the set is a simple function application of the data the set.
;;
;; All fns return a vector of a hypothesis identifier and the extension.
;;
(defn- tag
  "Create the name by which a hypothesis is known"
  ([n]
     (keyword (str n)))
  ([b n]
     (keyword (str (name b) "." n))))

;; ## Basic Hypotheses
(defn h-atom
  "Assume a single value"
  [n]
  [(tag n) (to-hyp [n])])

;; These are pretty self explanatory
(defn h-even []
  [:even (to-hyp (filter even? (range)))])

(defn  h-odd  []
  [:odd (to-hyp (filter odd? (range)))])

(defn h-mult [n]
  [(tag :mult n) (to-hyp (map #(* n %) (range)))])

(defn h-pow [n]
  [(tag :pow n) (to-hyp (map #(pow n %) (range)))])

(defn h-exp [n]
  [(tag :exp n) (to-hyp (map #(pow % n) (range)))])

(defn h-ends [n]
  [(tag :ends n) (to-hyp (range n max-num 10))])

(defn h-all []
  [:all (to-hyp (range))])

;; ### Composition Operators
;; These allow us to create compound hypotheses.

(defn union-h
  "Create an extension that is the union of two hypotheses"
  [[n1 e1] [n2 e2]]
  [(keyword (str (name n1) "_U_" (name n2))) (union e1 e2)])

(defn difference-h
  "Create an extension that is the difference of two hypotheses. ORDER MATTERS"
  [[n1 e1] [n2 e2]]
  [(keyword (str (name n1) "_U_" (name n2))) (difference e1 e2)])

;; ### Compound Hypotheses
;; These two should be written to be combinators, but lets not bother
(defn h-pow2-and [k]
  (union-h (h-pow 2) (h-atom k)))

(defn h-pow2-but [k]
  (difference-h (h-pow 2) (h-atom k)))


;; -----------------------------------------------------------------------------
;;  Inference functions

;; This is a basic likelihood function.  The idea is that it a closure
;; over all the parameters and can hence be passed around safely.
;; 1/|hypothesis|^|data| or 0.   Equation (3.2) pg 67.
(defn flat-likelihood [extension]
  (fn [data]
    (if (every? extension data)
      (pow (/ 1.0 (count extension)) (count data))
      0.0)))

;; Combine a likelihood function with a prior.
(defn generate-hypothesis [prior hyp-fn & param]
  (let [[id extension] (apply hyp-fn param)]
    {id {:likelihood-fn (flat-likelihood extension)
         :prior prior}}))

;; -----------------------------------------------------------------------------
;; ## A particular example

;; Generate the set of hypotheses given in the book pg 69
;; Note the priors are mapped in directly with the hypotheses-extensions
(defn hypotheses-set []
  (apply merge
   (concat
    [(generate-hypothesis 0.5 h-even)
     (generate-hypothesis 0.5 h-odd)
     (generate-hypothesis 0.1 h-exp 2)
     (generate-hypothesis 0.1 h-all)
     (generate-hypothesis 0.001 h-pow2-and 37)
     (generate-hypothesis 0.001 h-pow2-but 32)]
    (map (partial generate-hypothesis 0.1 h-mult) (range 3 11))
    (map (partial generate-hypothesis 0.1 h-ends) (range 1 10))
    (map (partial generate-hypothesis 0.1 h-pow)  (range 2 11)))))

;; No do some inference.

;; Start by creating an inference map, stating what we know.  That is
;; our hypotheses and some data.
(def partial-inference {:h    (hypotheses-set)
                        :data  [16]})

;; Feed that into the inference function, which returns the completed
;; inference map.
(def inference (inf/infer partial-inference))

;; Use Incanter to have a look at the likelihoods and posteriors.
(def i (:h inference))
(view (bar-chart (keys i) (map :likelihood (vals i)) :vertical false))
(view (bar-chart (keys i) (map :posterior (vals i))  :vertical false))

;; Now update the inference to reflect total disbelief in the :pow.4 hypothesis
;; and rerun.  Note that the inference will only update the nodes affected by this
;; change.  Its an example of inference-as-a-value.
(def another-inference (inf/infer (assoc-in inference [:h :pow.4 :prior] 0) inference))

;; And take a look at this.
(def i (:h another-inference))
(view (bar-chart (keys i) (map :likelihood (vals i)) :vertical false))
(view (bar-chart (keys i) (map :posterior (vals i))  :vertical false))

;; Have a look at the induced inference graph.  HAHAHA ITS HUUUUUUUUUGE.
;; You see is at the terminal with "dot -Tpng -o flow.png flow.dot" and
;; then "eog flow.png" or the equivalent

#_(com.stuartsierra.flow/write-dotfile  "flow.dot")
