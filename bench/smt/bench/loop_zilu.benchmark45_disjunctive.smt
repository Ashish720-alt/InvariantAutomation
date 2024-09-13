
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (or (> y 0) (> x 0)) (inv x y))
          (=> (> x 0) (=> (and (inv x y) (and (= xp (+ x 1)) (= yp y))) (inv xp yp)))
(=> (not (or  (> x 0))) (=> (<= x 0) (=> (and (inv x y) (and (= xp x) (= yp (+ y 1)))) (inv xp yp))))
          (=> (inv x y) (or (> x 0) (> y 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    