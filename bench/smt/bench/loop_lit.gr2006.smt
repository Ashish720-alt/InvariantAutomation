
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 0) (= y 0)) (inv x y))
          (=> (and (inv x y) (or (and (< x 50) (>= y -1)) (and (>= x 50) (>= y 1))) (< x 50)  (and (= xp (+ x 1)) (= yp (+ y 1)))) (inv xp yp))
(=> (and (inv x y) (or (and (< x 50) (>= y -1)) (and (>= x 50) (>= y 1))) (>= x 50) (not (or  (< x 50))) (and (= xp (+ x 1)) (= yp (+ y -1)))) (inv xp yp))
          (=> (and (not (or (and (< x 50) (>= y -1)) (and (>= x 50) (>= y 1)))) (inv x y)) (= x 100))
        )
      )
    )
    (check-sat)
    (get-model)
    