(set-logic HORN)

    (declare-fun |inv| (Int) Bool)
    (assert 
      (forall ((i Int) (ip Int)) 
        (and 
          (=> (and (< i 10) (> i -10)) (inv i))
          (=> (and (inv i) (and (< i 10) (> i -10)) (and (= ip (+ i 1)))) (inv ip))
          (=> (and (not (and (< i 10) (> i -10))) (inv i)) (= i 10))
        )
      )
    )
    (check-sat)
    (get-model)
    