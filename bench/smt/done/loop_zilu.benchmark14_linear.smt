
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((i Int) (ip Int)) 
        (and 
          (=> (and (>= i 0) (<= i 200)) (inv i))
          (=> (and (inv i) (> i 0) (and (= ip (+ i -1)))) (inv ip))
          (=> (and (not (> i 0)) (inv i)) (>= i 0))
        )
      )
    )
    (check-sat)
    (get-model)
    