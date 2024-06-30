
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (n Int) (xp Int) (yp Int) (np Int)) 
        (and 
          (=> (and (>= x 0) (<= (+ x (* -1 y)) 0) (< (+ y (* -1 n)) 0)) (inv x y n))
          (=> (> (+ x (* -1 y)) -1)(=> (and (inv x y n) (< (+ x (* -1 n)) 0) (and (= xp (+ x 1)) (= yp (+ y 1)) (= np n))) (inv xp yp np)))
(=> (<= (+ x (* -1 y)) -1)(=> (and (inv x y n) (< (+ x (* -1 n)) 0) (and (= xp (+ x 1)) (= yp y) (= np n))) (inv xp yp np)))
          (=> (and (not (< (+ x (* -1 n)) 0)) (inv x y n)) (= (+ y (* -1 n)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    