(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 0) (= y 0)) (inv x y))
          (=> (< x 500000) (=> (and (inv x y) (< x 1000000) (and (= xp (+ x 1)) (= yp (+ y 1)))) (inv xp yp)))
(=> (not (or  (< x 500000))) (=> (>= x 500000) (=> (and (inv x y) (< x 1000000) (and (= xp (+ x 1)) (= yp (+ y -1)))) (inv xp yp))))
          (=> (and (not (< x 1000000)) (inv x y)) (or (> y 0) (< y 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    