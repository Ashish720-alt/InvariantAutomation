(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 1) (= y 0)) (inv x y))
          (=> (and (inv x y) (< y 1024) (and (= xp 0) (= yp (+ y 1)))) (inv xp yp))
          (=> (and (not (< y 1024)) (inv x y)) (= x 0))
        )
      )
    )
    (check-sat)
    (get-model)
    