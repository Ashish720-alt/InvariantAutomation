
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((i Int) (ip Int)) 
        (and 
          (=> (= i 0) (inv i))
          (=> (and (inv i) (< i 6) (and (= ip (+ i 1)))) (inv ip))
          (=> (and (not (< i 6)) (inv i)) (<= i 6))
        )
      )
    )
    (check-sat)
    (get-model)
    