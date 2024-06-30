
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (or (> y 0) (> x 0)) (inv x y))
          (=> (> x 0)(=> (and (inv x y) (<= (+ x y) -2) (and (= xp (+ x 1)) (= yp y))) (inv xp yp)))
(=> (<= x 0)(=> (and (inv x y) (<= (+ x y) -2) (and (= xp x) (= yp (+ y 1)))) (inv xp yp)))
          (=> (and (not (<= (+ x y) -2)) (inv x y)) (or (> y 0) (> x 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    