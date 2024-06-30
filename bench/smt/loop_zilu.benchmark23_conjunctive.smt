
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (ip Int) (jp Int)) 
        (and 
          (=> (and (= i 0) (= j 0)) (inv i j))
          (=> (and (inv i j) (< i 100) (and (= ip (+ i 1)) (= jp (+ j 2)))) (inv ip jp))
          (=> (and (not (< i 100)) (inv i j)) (= j 200))
        )
      )
    )
    (check-sat)
    (get-model)
    