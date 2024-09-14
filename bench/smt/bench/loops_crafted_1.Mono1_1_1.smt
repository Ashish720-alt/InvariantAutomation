
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (= x 0) (inv x))
          (=> (and (inv x) (< x 100000000) (< x 10000000)  (and (= xp (+ x 1)))) (inv xp))
(=> (and (inv x) (< x 100000000) (>= x 10000000) (not (or  (< x 10000000))) (and (= xp (+ x 2)))) (inv xp))
          (=> (and (not (< x 100000000)) (inv x)) (= x 100000001))
        )
      )
    )
    (check-sat)
    (get-model)
    