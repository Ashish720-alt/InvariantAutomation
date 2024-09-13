
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= (+ x (* -1 y)) 0) (>= y 0)) (inv x y))
          (=> (and (inv x y) (or (< y 0) (> y 0)) (and (= xp (+ x -1)) (= yp (+ y -1)))) (inv xp yp))
          (=> (and (not (or (< y 0) (> y 0))) (inv x y)) (= y 0))
        )
      )
    )
    (check-sat)
    (get-model)
    