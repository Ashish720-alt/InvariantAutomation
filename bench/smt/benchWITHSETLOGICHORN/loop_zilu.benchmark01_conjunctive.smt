(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 1) (= y 1)) (inv x y))
          (=> (and (inv x y) (and (= xp (+ x y)) (= yp (+ x y)))) (inv xp yp))
          (=> (inv x y) (>= y 1))
        )
      )
    )
    (check-sat)
    (get-model)
    