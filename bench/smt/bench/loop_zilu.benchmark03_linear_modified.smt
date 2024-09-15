
    (declare-fun |inv| (Int Int Int Int) Bool)
    (assert 
      (forall ((x Int) (y Int) (i Int) (j Int) (xp Int) (yp Int) (ip Int) (jp Int)) 
        (and 
          (=> (and (= x 0) (= y 0) (= i 0) (= j 0)) (inv x y i j))
          (=> (and (inv x y i j)    (or (and (= xp (+ x 1)) (= yp (+ y 1)) (= ip (+ (+ x i) 1)) (= jp (+ (+ y j) 1))) (and (= xp (+ x 1)) (= yp (+ y 1)) (= ip (+ (+ x i) 1)) (= jp (+ (+ y j) 2))))) (inv xp yp ip jp))
          (=> (inv x y i j) (>= (+ (* -1 i) j) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    