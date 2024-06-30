
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((i Int) (c Int) (ip Int) (cp Int)) 
        (and 
          (=> (and (= c 0) (= i 0)) (inv i c))
          (=> (and (inv i c) (and (< i 100) (> i -1)) (and (= ip (+ i 1)) (= cp (+ i c)))) (inv ip cp))
          (=> (and (not (and (< i 100) (> i -1))) (inv i c)) (>= c 0))
        )
      )
    )
    (check-sat)
    (get-model)
    