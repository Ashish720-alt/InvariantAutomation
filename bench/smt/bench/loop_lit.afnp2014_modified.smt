
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 2) (= y 2)) (inv x y))
          (=> (and (inv x y) (< y 1000)   (and (= xp (+ x y)) (= yp (+ y 1)))) (inv xp yp))
          (=> (and (not (< y 1000)) (inv x y)) (>= (+ x (* -1 y)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    