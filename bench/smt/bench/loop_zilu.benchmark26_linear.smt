
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (xp Int) (yp Int)) 
        (and 
          (=> (< (+ x (* -1 y)) 0) (inv x y))
          (=> (and (inv x y) (< (+ x (* -1 y)) 0) (and (= xp (+ x 1)) (= yp y))) (inv xp yp))
          (=> (and (not (< (+ x (* -1 y)) 0)) (inv x y)) (= (+ x (* -1 y)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    