
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (ip Int) (jp Int)) 
        (and 
          (=> (and (= i 1) (= j 10)) (inv i j))
          (=> (and (inv i j) (>= (+ (* -1 i) j) 0) (and (= ip (+ i 2)) (= jp (+ j -1)))) (inv ip jp))
          (=> (and (not (>= (+ (* -1 i) j) 0)) (inv i j)) (or (= j 6) (>= (+ (* -1 i) j) 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    