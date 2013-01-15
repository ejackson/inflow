;; This is the core inference namespace, the main function of which is infer
;;
;; ## Data structures
;; There are two main structures, and inference- and a flow-map.
;; An inference is a map with a key signature like
;;
;;     {:norm,
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
(defn flat-map
  "Return a flat map representation of a nested map.  Nested keys return as vectors."
  ([m] (flat-map {} [] m))
  ([a ks m]
     (if (map? m)
       (reduce into (map (fn [[k v]] (flat-map a (conj ks k) v)) (seq m)))
       (assoc a ks m))))

(defn nested-map
  "Return a nested map representation of a flat map."
  [m]
  (reduce (fn [m [ks v]] (assoc-in m ks v)) {} m))

;; -----------------------------------------------------------------------------
;; ## Keys

;; The flow representation is flat these are the names we use to unmap the nested
;; inference-map to the flow-map node names
;;
(defn k-cat [s1 s2]
  (keyword (str s1 (name s2))))

(defn prior-key [id]
  (k-cat "prior-" id))

(defn likelihood-fn-key [id]
  (k-cat "likelihood-fn-" id))

(defn likelihood-key [id]
  (k-cat "likelihood-" id))

(defn unnorm-posterior-key [id]
  (k-cat "unnorm-posterior-" id))

(defn posterior-key [id]
  (k-cat "posterior-" id))

;;--------------------------------------------------------------------
;; ## Conversions

;; ### flow->
;; Convert from a flow to an inference map.  Its a bit ugly as the ids
;; are distributed amoung the keys, so we have to munge it a lot.
(defn- inf-keys [id]
  {(prior-key id)            :prior
   (likelihood-fn-key id)    :likelihood-fn
   (likelihood-key id)       :likelihood
   (unnorm-posterior-key id) :unnorm-posterior
   (posterior-key id)        :posterior})

;; Normalise the names of all the hypotheses coming out of the complete graph
(defn- extract-hypotheses-keys [id inf-flow]
  (let [key-xf (inf-keys id)]
    {id (-> inf-flow
            (select-keys (keys key-xf))
            (rename-keys key-xf))}))

;; A couple of helper fn to extract keys from a flow.  The game is to know
;; that every hypothesis must have a prior.  So find all the keys in the flow
;; that relate to priors and pull off that section of the name which is the
;; id.  Its all rather ugly.
;;
(defn- id-from-prior-key [k]
  (second (split (str (name k)) #"prior-")))

(defn- hyp-ids
  [flow]
  (->> (keys flow)
       (map id-from-prior-key)
       (filter identity)
       (map keyword)))

(defn flow->
  "Takes a flow and turns it into an inference map"
  [{:keys [norm data] :as inf}]
  {:norm norm
   :data data
   :hypotheses
   (reduce
    (fn [m id]
      (merge m (extract-hypotheses-keys id inf)))
    {}
    (hyp-ids inf))})

;; -----------------------------------------------------------------------------
;; ### ->flow

;;  Convert an inference map to a flow map.  Note that there is
;;  no metadata here, as the keys cannot necessarily support it.
;;  If you want that, you need to create a generative flow.
;;
(defn- flatten-hyp
  "Flatten an individual hypothesis to its flow representation."
  [id hypothesis]
  (reduce
   (fn [m [f-k i-k]]
     (if (i-k hypothesis)
       (assoc m f-k (i-k hypothesis))
       m))
   {}
   (inf-keys id)))

(defn ->flow
  "Turn an inference into a flow.  If data and norm are given they are included."
  [{:keys [hypotheses] :as inf}]
  (merge
   (select-keys inf [:data :norm])
   (reduce (fn [f [id h]] (merge f (flatten-hyp id h)))
           (flow/flow)
           hypotheses)))

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
(defn- prior-flow-fn [prior] (flow/flow-fn [] prior))

(defn- likelihood-fn-flow-fn [likelihood-fn] (flow/flow-fn [] likelihood-fn))

(defn- likelihood-flow-fn [id]
  (flow/with-inputs [(likelihood-fn-key id) :data]
    (fn [{data :data :as r}]
      (((likelihood-fn-key id) r) data))))

(defn- unnorm-posterior-flow-fn [id]
  (flow/with-inputs [(likelihood-key id) (prior-key id)]
    (fn [r]
      (* ((likelihood-key id) r) ((prior-key id) r)))))

(defn- posterior-flow-fn [id]
  (flow/with-inputs [(unnorm-posterior-key id) :norm]
    (fn [r]
      (/ ((unnorm-posterior-key id) r) (:norm r)))))

;; ### Composition
;; Create a Flow that combines all the flow functions.

(defn- add-hypothesis
  "Add the flow function of a single hypothesis to the flow"
  [flow [id {:keys [likelihood-fn prior] :as hypothesis}]]
  (-> flow
      (assoc (prior-key id)            (prior-flow-fn prior))
      (assoc (likelihood-fn-key id)    (likelihood-fn-flow-fn likelihood-fn))
      (assoc (likelihood-key id)       (likelihood-flow-fn id))
      (assoc (unnorm-posterior-key id) (unnorm-posterior-flow-fn id))
      (assoc (posterior-key id)        (posterior-flow-fn id))))

;; This is a slightly different node as it has inter-hypotheses dependencies.
(defn- normalise-flow
  "Add the normalisation node, dep'd on all the unnorm-posterior nodes"
  [flow hypotheses]
  (let [post-nodes (mapv unnorm-posterior-key (keys hypotheses))]
    (assoc flow :norm (flow/with-inputs post-nodes
                        (fn [r] (apply + (vals (select-keys r post-nodes))))))))

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
  [{:keys [hypotheses] :as new-inf} old-inf]
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
  ([{:keys [hypotheses] :as new-inference} given-inference]
     (flow->
      (flow/run (generative-flow hypotheses) (untaint new-inference given-inference)))))
