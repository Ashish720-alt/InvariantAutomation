
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (z Int) (xp Int) (yp Int) (zp Int)) 
        (and 
          (=> (and (= x 0) (= y 10000000) (= z 5000000)) (inv x y z))
          (=> (>= x 5000000)(=> (and (inv x y z) (< (+ x (* -1 y)) 0) (and (= xp (+ x 1)) (= yp y) (= zp (+ z -1)))) (inv xp yp zp)))
(=> (< x 5000000)(=> (and (inv x y z) (< (+ x (* -1 y)) 0) (and (= xp (+ x 1)) (= yp y) (= zp z))) (inv xp yp zp)))
          (=> (and (not (< (+ x (* -1 y)) 0)) (inv x y z)) (and (> x 0) (< x 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    