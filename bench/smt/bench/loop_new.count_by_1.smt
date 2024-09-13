
    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((i Int) (ip Int)) 
        (and 
          (=> (= i 0) (inv i))
          (=> (and (inv i) (< i 1000000) (and (= ip (+ i 1)))) (inv ip))
          (=> (and (not (< i 1000000)) (inv i)) (= i 1000000))
        )
      )
    )
    (check-sat)
    (get-model)
    