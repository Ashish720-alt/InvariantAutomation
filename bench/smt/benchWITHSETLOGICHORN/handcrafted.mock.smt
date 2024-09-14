(set-logic HORN)

    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (= x 0) (inv x))
          (=> (and (inv x) (<= x 5) (and (= xp (+ x 1)))) (inv xp))
          (=> (and (not (<= x 5)) (inv x)) (<= x 6))
        )
      )
    )
    (check-sat)
    (get-model)
    