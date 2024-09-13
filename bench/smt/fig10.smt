    (declare-var x Int)
    (declare-var xp Int)
    (declare-var y Int)
    (declare-var yp Int)
    (declare-var N Int)

    (assert 
       (and 
        (> N 10)
      ;  (not (=> (and (= x 0) (= y 1)) (or (<= x (* 2 N)) (>= y N))))
      ;  (not (=> (and (or (<= x (* 2 N)) (>= y N)) (< x (* 10 N)) (< x N) (= xp (+ x y)) (= yp y)) (or (<= xp (* 2 N)) (>= yp N))))
        (not (=> (and (or (<= x (* 2 N)) (>= y N)) (< x (* 10 N)) (>= x N) (= xp (- x N)) (= yp (+ y 1))) (or (<= xp (* 2 N)) (>= yp N))))
      ;  (not (=> (and (not (< x (* 10 N))) (or (<= x (* 2 N)) (>= y N))) (>= y N)))
       )
      )
    (check-sat)
    (get-model)
