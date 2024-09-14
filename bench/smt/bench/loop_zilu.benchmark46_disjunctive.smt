
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (z Int) (xp Int) (yp Int) (zp Int)) 
        (and 
          (=> (or (> y 0) (> x 0) (> z 0)) (inv x y z))
          (=> (and (inv x y z)  (and (> x 0) (> y 0))  (and (= xp (+ x 1)) (= yp (+ y 1)) (= zp z))) (inv xp yp zp))
(=> (and (inv x y z)  (and (> x 0) (<= y 0)) (not (or  (and (> x 0) (> y 0)))) (and (= xp (+ x 1)) (= yp y) (= zp (+ z 1)))) (inv xp yp zp))
(=> (and (inv x y z)  (and (<= x 0) (> y 0)) (not (or  (and (> x 0) (> y 0)) (and (> x 0) (<= y 0)))) (and (= xp x) (= yp (+ y 1)) (= zp z))) (inv xp yp zp))
(=> (and (inv x y z)  (and (<= x 0) (<= y 0)) (not (or  (and (> x 0) (> y 0)) (and (> x 0) (<= y 0)) (and (<= x 0) (> y 0)))) (and (= xp x) (= yp y) (= zp (+ z 1)))) (inv xp yp zp))
          (=> (inv x y z) (or (> y 0) (> x 0) (> z 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    