
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((lo Int) (mid Int) (hi Int) (lop Int) (midp Int) (hip Int)) 
        (and 
          (=> (and (= lo 0) (> mid 0) (< mid 1000000) (= (+ (* -2 mid) hi) 0)) (inv lo mid hi))
          (=> (and (inv lo mid hi) (> mid 0)   (and (= lop (+ lo 1)) (= midp (+ mid -1)) (= hip (+ hi -1)))) (inv lop midp hip))
          (=> (and (not (> mid 0)) (inv lo mid hi)) (= (+ lo (* -1 hi)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    