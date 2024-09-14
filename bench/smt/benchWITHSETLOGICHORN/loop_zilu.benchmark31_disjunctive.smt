(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (< x 0) (inv x y))
          (=> (and (inv x y) (< x 0) (and (= xp (+ x y)) (= yp (+ y 1)))) (inv xp yp))
          (=> (and (not (< x 0)) (inv x y)) (>= y 0))
        )
      )
    )
    (check-sat)
    (get-model)
    