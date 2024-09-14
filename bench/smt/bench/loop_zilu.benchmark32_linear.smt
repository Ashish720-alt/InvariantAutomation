
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (or (= x 1) (= x 2)) (inv x))
          (=> (and (inv x)  (= x 1)  (and (= xp (+ 0 2)))) (inv xp))
(=> (and (inv x)  (= x 2) (not (or  (= x 1))) (and (= xp (+ 0 1)))) (inv xp))
(=> (and (inv x)  (or (<= x 0) (>= x 3)) (not (or  (= x 1) (= x 2))) (and (= xp x))) (inv xp))
          (=> (inv x) (<= x 8))
        )
      )
    )
    (check-sat)
    (get-model)
    