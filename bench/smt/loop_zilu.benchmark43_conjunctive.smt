
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (< x 100) (< y 100)) (inv x y))
          (=> (and (inv x y) (and (< x 100) (< y 100)) (and (= xp (+ x 1)) (= yp (+ y 1)))) (inv xp yp))
          (=> (and (not (and (< x 100) (< y 100))) (inv x y)) (or (= x 100) (= y 100)))
        )
      )
    )
    (check-sat)
    (get-model)
    