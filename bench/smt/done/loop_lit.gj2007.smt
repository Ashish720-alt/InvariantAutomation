
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 0) (= y 50)) (inv x y))
          (=> (< x 50) (=> (and (inv x y) (< x 100) (and (= xp (+ x 1)) (= yp y))) (inv xp yp)))
(=> (not (or  (< x 50))) (=> (>= x 50) (=> (and (inv x y) (< x 100) (and (= xp (+ x 1)) (= yp (+ y 1)))) (inv xp yp))))
          (=> (and (not (< x 100)) (inv x y)) (= y 100))
        )
      )
    )
    (check-sat)
    (get-model)
    