
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= (+ x (* -1 y)) 0) (= y 0)) (inv x y))
          (=> (and (inv x y) (and (= xp (+ x 4)) (= yp (+ y 1)))) (inv xp yp))
          (=> (inv x y) (and (= (+ x (* -4 y)) 0) (>= x 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    