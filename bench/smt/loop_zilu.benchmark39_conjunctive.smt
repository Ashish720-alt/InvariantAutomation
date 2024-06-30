
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= (+ x (* -4 y)) 0) (>= x 0)) (inv x y))
          (=> (and (inv x y) (> x 0) (and (= xp (+ x -4)) (= yp (+ y -1)))) (inv xp yp))
          (=> (and (not (> x 0)) (inv x y)) (>= y 0))
        )
      )
    )
    (check-sat)
    (get-model)
    