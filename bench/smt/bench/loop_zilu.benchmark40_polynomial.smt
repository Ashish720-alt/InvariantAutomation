
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (<= x 0) (<= y 0)) (inv x y))
          (=> (and (inv x y)  (> x 0)  (and (= xp x) (= yp (+ y 1)))) (inv xp yp))
(=> (and (inv x y)  (< x 0) (not (or  (> x 0))) (and (= xp (+ x -1)) (= yp y))) (inv xp yp))
(=> (and (inv x y)  (and (= x 0) (> y 0)) (not (or  (> x 0) (< x 0))) (and (= xp (+ x 1)) (= yp y))) (inv xp yp))
(=> (and (inv x y)  (and (= x 0) (<= y 0)) (not (or  (> x 0) (< x 0) (and (= x 0) (> y 0)))) (and (= xp (+ x -1)) (= yp y))) (inv xp yp))
          (=> (inv x y) (and (<= x 0) (<= y 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    