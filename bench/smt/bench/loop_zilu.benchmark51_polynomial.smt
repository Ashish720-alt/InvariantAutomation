
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (and (>= x 0) (<= x 50)) (inv x))
          (=> (and (inv x)  (or (= x 0) (> x 50))  (and (= xp (+ x 1)))) (inv xp))
(=> (and (inv x)  (or (< x 0) (and (> x 0) (<= x 50))) (not (or  (or (= x 0) (> x 50)))) (and (= xp (+ x -1)))) (inv xp))
          (=> (inv x) (and (>= x 0) (<= x 50)))
        )
      )
    )
    (check-sat)
    (get-model)
    