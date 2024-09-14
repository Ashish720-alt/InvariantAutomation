(set-logic HORN)

    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (and (>= x 0) (<= x 50)) (inv x))
          (=> (or (= x 0) (> x 50)) (=> (and (inv x) (and (= xp (+ x 1)))) (inv xp)))
(=> (not (or  (or (= x 0) (> x 50)))) (=> (or (< x 0) (and (> x 0) (<= x 50))) (=> (and (inv x) (and (= xp (+ x -1)))) (inv xp))))
          (=> (inv x) (and (>= x 0) (<= x 50)))
        )
      )
    )
    (check-sat)
    (get-model)
    