
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (= x 0) (inv x))
          (=> (< x 10000000) (=> (and (inv x) (< x 100000000) (and (= xp (+ x 1)))) (inv xp)))
(=> (not (or  (< x 10000000))) (=> (>= x 10000000) (=> (and (inv x) (< x 100000000) (and (= xp (+ x 2)))) (inv xp))))
          (=> (and (not (< x 100000000)) (inv x)) (= x 100000001))
        )
      )
    )
    (check-sat)
    (get-model)
    