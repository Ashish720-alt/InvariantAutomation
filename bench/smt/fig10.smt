(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((N Int) (x Int) (xp Int) (y Int) (yp Int)) 
        (and 
          (= N 100000000)
          (=> (and (= x 0) (= y 1)) (inv x y))
          (=> (and (inv x y) (< x (* 10 N)) (and (=> (< x N) (and (= xp (+ x y)) (= yp y)))  (=> (>= x N) (and (= xp (- x N)) (= yp (+ y 1))) ))) 
                                                                                                      (inv xp yp))
          (=> (and (not (< x (* 10 N))) (inv x y)) (>= y N))
        )
      )
    )
    (check-sat)

    