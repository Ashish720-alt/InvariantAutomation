(set-logic HORN)

    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (z Int) (xp Int) (yp Int) (zp Int)) 
        (and 
          (=> (and (= (+ x (* -1 y)) 0) (>= x 0) (= (+ x (+ y z)) 0)) (inv x y z))
          (=> (and (inv x y z) (> x 0) (and (= xp (+ x -1)) (= yp (+ y -1)) (= zp (+ z 2)))) (inv xp yp zp))
          (=> (and (not (> x 0)) (inv x y z)) (<= z 0))
        )
      )
    )
    (check-sat)
    (get-model)
    