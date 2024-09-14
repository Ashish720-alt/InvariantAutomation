(set-logic HORN)

    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (>= x 0) (inv x))
          (=> (and (inv x) (and (< x 100) (>= x 0)) (and (= xp (+ x 1)))) (inv xp))
          (=> (and (not (and (< x 100) (>= x 0))) (inv x)) (>= x 100))
        )
      )
    )
    (check-sat)
    (get-model)
    