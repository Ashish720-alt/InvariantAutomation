(set-logic HORN)

    (declare-fun |inv| (Int Int Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (x Int) (y Int) (ip Int) (jp Int) (xp Int) (yp Int)) 
        (and 
          (=> (and (>= i 0) (>= j 0) (= (+ (* -1 i) x) 0) (= (+ (* -1 j) y) 0)) (inv i j x y))
          (=> (and (inv i j x y) (or (> x 0) (< x 0)) (and (= ip i) (= jp j) (= xp (+ x -1)) (= yp (+ y -1)))) (inv ip jp xp yp))
          (=> (and (not (or (> x 0) (< x 0))) (inv i j x y)) (or (< (+ i (* -1 j)) 0) (> (+ i (* -1 j)) 0) (= y 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    