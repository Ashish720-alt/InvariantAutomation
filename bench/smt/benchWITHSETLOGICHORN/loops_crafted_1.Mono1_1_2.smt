(set-logic HORN)

    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (= x 0) (inv x))
          (=> (< x 1000000) (=> (and (inv x) (< x 1000000) (and (= xp (+ x 1)))) (inv xp)))
(=> (not (or  (< x 1000000))) (=> (>= x 1000000) (=> (and (inv x) (< x 1000000) (and (= xp (+ x 2)))) (inv xp))))
          (=> (and (not (< x 1000000)) (inv x)) (= x 1000000))
        )
      )
    )
    (check-sat)
    (get-model)
    