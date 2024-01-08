(ns dev.user
  (:require [nextjournal.clerk :as clerk]))

(defn run [_]
  (clerk/serve! {:browse? true
                 :watch-paths ["notebooks"]})
  (println "If your browser has not opened then please navigate to
https://localhost/7777"))

(comment
  (clerk/show! 'nextjournal.clerk.tap)
  (clerk/show! "notebooks/part3.clj"))
