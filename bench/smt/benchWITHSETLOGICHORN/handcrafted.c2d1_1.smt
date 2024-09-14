(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (= x 1) (= y 1)) (inv x y))
        )
      )
    )

     (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (inv x y) (and (= xp (+ x y)) (= yp (+ y 1)))) (inv xp yp))

        )
      )
    )

     (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (inv x y) (and (>= (+ x (* -1 y)) 0) (>= y 1)))
        )
      )
    )      
    (check-sat)
    (get-model)
    (exit)
    