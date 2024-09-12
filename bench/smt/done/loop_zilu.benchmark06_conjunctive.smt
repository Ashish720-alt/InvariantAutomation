
    (declare-fun |inv| (Int Int Int Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (x Int) (y Int) (k Int) (ip Int) (jp Int) (xp Int) (yp Int) (kp Int)) 
        (and 
          (=> (and (= (+ x (+ y (* -1 k))) 0) (= j 0)) (inv i j x y k))
          (=> (= (+ (* -1 i) j) 0) (=> (and (inv i j x y k) (and (= ip i) (= jp (+ j 1)) (= xp (+ x 1)) (= yp (+ y -1)) (= kp k))) (inv ip jp xp yp kp)))
(=> (not (or  (= (+ (* -1 i) j) 0))) (=> (or (< (+ (* -1 i) j) 0) (< (+ i (* -1 j)) 0)) (=> (and (inv i j x y k) (and (= ip i) (= jp (+ j 1)) (= xp (+ x -1)) (= yp (+ y 1)) (= kp k))) (inv ip jp xp yp kp))))
          (=> (inv i j x y k) (= (+ x (+ y (* -1 k))) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    