;; This is the core inference namespace, the main function of which is infer
;;
;; ## Data structures
;; There are two main structures, and inference- and a flow-map.
;; An inference is a map with a key signature like
;;
;;     {:post-norm,
;;      :data,
;;      :hypotheses {:h-id {:prior, :likelihood-fn,
;;                          :likelihood, :unnorm-posterior, :posterior},
;;                   ...}}
;;
;; it is something we would pass around to prediction algorithms etc.
;;
;; A flow-map is just a flattened version of the inference-map structured
;; so that we can pass it into Flow for computation.
;;
;; ## Main Sections
;; 1. Conversion: The functions ->flow and flow-> are to convert between the two main structres
;; 2. Flow Generation: Populate a flow structure with the functions needed to make it work
;; 3. Flow Differencing: We treat inferences as values and can do updates. This section works out
;;      what has changed in a flow so that a minimal update is possible.
;; 4. Inference: Run the thing!
;;
(ns inflow.core
  (:use
   [clojure.set     :only [rename-keys difference intersection union]]
   [clojure.string  :only [split]])
  (:require
   [com.stuartsierra.flow :as flow]
   [clojure.tools.namespace.dependency :as dep]))

;; -----------------------------------------------------------------------------
;;  Map flattening and raising
(defn ->flow
  "Return a flat map representation of a nested map.  Nested keys return as vectors."
  ([m] (->flow {} [] m))
  ([a [f s :as ks] m]
       (if (map? m)
         (reduce into {} (map (fn [[k v]] (->flow a (conj ks k) v)) (seq m)))
         (assoc a (if s ks f) m))))

(defn flow->
  "Return a nested map representation of a flat map."
  [m]
  (reduce (fn [m [ks v]] (assoc-in m (if (vector? ks) ks [ks]) v)) {} m))

;; -----------------------------------------------------------------------------
;; ## Generative Flow Creation
;; A generative flow is flow comprising of inference performing functions, whereas
;; a flow-map is just the inputs or outputs of such a process.  The main function
;; here is generative-flow

;; ###Flow functions
;; Each step of the inference is the same across hypotheses.  This collection
;; of functions performs those steps, and in addition, decorates each function
;; with metadata describing the dependency graph that they form.
;;
;; Notice that the dependency graph is described *only* here
;;
(defn- unnorm-prior-flow-fn [prior] (flow/flow-fn [] prior))

(defn- prior-flow-fn [id]
  (flow/with-inputs [[:h id :unnorm-prior] :prior-norm]
    (fn [r]
      (/ (r [:h id :unnorm-prior]) (:prior-norm r)))))

(defn- likelihood-fn-flow-fn [likelihood-fn] (flow/flow-fn [] likelihood-fn))

(defn- likelihood-flow-fn [id]
  (flow/with-inputs [[:h id :likelihood-fn] :data]
    (fn [{data :data :as r}]
      ((r [:h id :likelihood-fn]) data))))

(defn- unnorm-posterior-flow-fn [id]
  (flow/with-inputs [[:h id :likelihood] [:h id :prior]]
    (fn [r]
      (* (r [:h id :likelihood]) (r [:h id :prior])))))

(defn- posterior-flow-fn [id]
  (flow/with-inputs [[:h id :unnorm-posterior] :post-norm]
    (fn [r]
      (/ (r [:h id :unnorm-posterior]) (:post-norm r)))))

;; ### Composition
;; Create a Flow that combines all the flow functions.

(defn- add-hypothesis
  "Add the flow function of a single hypothesis to the flow"
  [flow [id {:keys [likelihood-fn prior] :as hypothesis}]]
  (-> flow
      (assoc [:h id :unnorm-prior]     (unnorm-prior-flow-fn prior))
      (assoc [:h id :prior]            (prior-flow-fn id))
      (assoc [:h id :likelihood-fn]    (likelihood-fn-flow-fn likelihood-fn))
      (assoc [:h id :likelihood]       (likelihood-flow-fn id))
      (assoc [:h id :unnorm-posterior] (unnorm-posterior-flow-fn id))
      (assoc [:h id :posterior]        (posterior-flow-fn id))))

;; This is a slightly different node as it has inter-hypotheses dependencies.
(defn- normalise-flow
  "Add the normalisation node, dep'd on all the unnorm-posterior nodes"
  [flow hypotheses]
  (let [post-nodes  (map (fn [id] [:h id :unnorm-posterior]) (keys hypotheses))
        prior-nodes (map (fn [id] [:h id :unnorm-prior])     (keys hypotheses))]
    (-> flow
        (assoc :prior-norm (flow/with-inputs prior-nodes
                             (fn [r] (apply + (vals (select-keys r prior-nodes))))))
        (assoc :post-norm (flow/with-inputs post-nodes
                            (fn [r] (apply + (vals (select-keys r post-nodes)))))))))

(defn generative-flow
  "Given a set of hypotheses, return a flow that can generate all the outputs"
  [hypotheses]
  (reduce add-hypothesis (normalise-flow (flow/flow) hypotheses) hypotheses))


;; -----------------------------------------------------------------------------
;; ## Flow Differencing
;; A useful feature is to be able to take 'given' data, and perform an inference
;; based upon it.  A typical example would be to perform an inference, then modify
;; the prior of one of the hypotheses.  You don't want to have to recalculate the
;; entire inference, but only those nodes affected by the changed prior.  This
;; section is the backbone of this notion.  It works out which nodes have changed and
;; the set of their transient dependencies.

(defn- node-differences
  "Return all the nodes that have been added, removed or changed between m1 and m2"
  [m1 m2]
  (let [ks1 (set (keys m1))
        ks2 (set (keys m2))
        ks1*ks2 (intersection ks1 ks2)]
    (union
     (into #{} (remove (fn [k] (= (m1 k) (m2 k))) ks1*ks2))
     (difference ks1 ks2)
     (difference ks2 ks1))))

(defn- taint-nodes
  "Return all the nodes in the flow that are affected by changes in changed-nodes."
  [flow changed-nodes]
  (into #{}
        (mapcat (partial dep/transitive-dependents
                         (#'com.stuartsierra.flow/flow-graph flow))
                changed-nodes)))

(defn untaint
  "Given two inferences, where the old inference is consistent, discover all
   the nodes in the new inference that differ from the old, and their dependents.
   Return a flow with these dependents removed.  This can be used as the 'given'
   inference in infer as the remaining graph is consistent."
  [{hypotheses :h :as new-inf} old-inf]
  (let [old-flow       (->flow old-inf)
        new-flow       (->flow new-inf)
        new-gen-flow   (generative-flow hypotheses)
        changed-nodes  (node-differences new-flow old-flow)
        tainted-nodes  (taint-nodes new-gen-flow changed-nodes)]
    (apply dissoc new-flow tainted-nodes)))


;; -----------------------------------------------------------------
;; ## Inference
;; The main deal.
(defn infer
  "Accepts an partial inference map, and returns a full inference map containing
   all the missing downstream values.  Optionally include an old-inf map containing
   a previous inference, then infer will only do an update calculation for those
   nodes that need to be updated on account of changes."
  ([new-inference] (infer new-inference {}))
  ([{hypotheses :h :as new-inference} given-inference]
     (flow->
      (flow/run (generative-flow hypotheses) (untaint new-inference given-inference)))))
