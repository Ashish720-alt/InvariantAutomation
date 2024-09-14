
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (= x 0) (inv x))
          (=> (and (inv x) (< x 1000000) (< x 1000000)  (and (= xp (+ x 1)))) (inv xp))
(=> (and (inv x) (< x 1000000) (>= x 1000000) (not (or  (< x 1000000))) (and (= xp (+ x 2)))) (inv xp))
          (=> (and (not (< x 1000000)) (inv x)) (= x 1000000))
        )
      )
    )
    (check-sat)
    (get-model)
    