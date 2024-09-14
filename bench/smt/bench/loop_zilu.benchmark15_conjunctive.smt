
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((low Int) (mid Int) (high Int) (lowp Int) (midp Int) (highp Int)) 
        (and 
          (=> (and (= low 0) (>= mid 1) (= (+ (* -2 mid) high) 0)) (inv low mid high))
          (=> (and (inv low mid high) (> mid 0)   (and (= lowp (+ low 1)) (= midp (+ mid -1)) (= highp (+ high -1)))) (inv lowp midp highp))
          (=> (and (not (> mid 0)) (inv low mid high)) (= (+ low (* -1 high)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    