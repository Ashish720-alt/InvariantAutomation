
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((x Int) (xp Int)) 
        (and 
          (=> (or (= x 1) (= x 2)) (inv x))
          (=> (= x 1) (=> (and (inv x) (and (= xp (+ 0 2)))) (inv xp)))
(=> (not (or  (= x 1))) (=> (= x 2) (=> (and (inv x) (and (= xp (+ 0 1)))) (inv xp))))
(=> (not (or  (= x 1) (= x 2))) (=> (or (<= x 0) (>= x 3)) (=> (and (inv x) (and (= xp x))) (inv xp))))
          (=> (inv x) (<= x 8))
        )
      )
    )
    (check-sat)
    (get-model)
    