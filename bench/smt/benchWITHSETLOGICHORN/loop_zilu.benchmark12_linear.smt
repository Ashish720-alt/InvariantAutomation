(set-logic HORN)

    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (t Int) (xp Int) (yp Int) (tp Int)) 
        (and 
          (=> (or (and (> (+ x (* -1 y)) 0) (= (+ y (* -1 t)) 0)) (and (< (+ x (* -1 y)) 0) (= (+ y (* -1 t)) 0))) (inv x y t))
          (=> (> x 0) (=> (and (inv x y t) (and (= xp x) (= yp (+ x y)) (= tp t))) (inv xp yp tp)))
(=> (not (or  (> x 0))) (=> (<= x 0) (=> (and (inv x y t) (and (= xp x) (= yp y) (= tp t))) (inv xp yp tp))))
          (=> (inv x y t) (>= (+ y (* -1 t)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    