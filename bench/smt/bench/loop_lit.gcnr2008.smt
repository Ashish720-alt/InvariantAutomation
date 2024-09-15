
    (declare-fun |inv| (Int Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (z Int) (w Int) (xp Int) (yp Int) (zp Int) (wp Int)) 
        (and 
          (=> (and (= x 0) (= y 0) (= z 0) (= w 0)) (inv x y z w))
          (=> (and (inv x y z w) (< y 10000) (>= x 4)  (or (and (= xp (+ x 1)) (= yp (+ y 100)) (= zp (+ z 10)) (= wp (+ w 1))) (and (= xp (+ x 1)) (= yp (+ y 1)) (= zp (+ z 10)) (= wp (+ w 1))))) (inv xp yp zp wp))
(=> (and (inv x y z w) (< y 10000) (and (>= (+ (* -100 x) z) 0) (< x 4) (> (+ y (* -10 w)) 0)) (not (or  (>= x 4))) (or (and (= xp (+ x 1)) (= yp (+ y 100)) (= zp (+ z 10)) (= wp (+ w 1))) (and (= xp x) (= yp (* -1 y)) (= zp (+ z 10)) (= wp (+ w 1))))) (inv xp yp zp wp))
(=> (and (inv x y z w) (< y 10000) (or (and (< x 4) (<= (+ y (* -10 w)) 0)) (and (< x 4) (< (+ (* -100 x) z) 0)) (and (< x 4) (>= x 4))) (not (or  (>= x 4) (and (>= (+ (* -100 x) z) 0) (< x 4) (> (+ y (* -10 w)) 0)))) (or (and (= xp (+ x 1)) (= yp (+ y 100)) (= zp (+ z 10)) (= wp (+ w 1))) (and (= xp x) (= yp y) (= zp (+ z 10)) (= wp (+ w 1))))) (inv xp yp zp wp))
          (=> (and (not (< y 10000)) (inv x y z w)) (and (>= x 4) (<= y 2)))
        )
      )
    )
    (check-sat)
    (get-model)
    