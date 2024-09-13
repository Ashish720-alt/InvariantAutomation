
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((n Int) (sum Int) (i Int) (np Int) (sump Int) (ip Int)) 
        (and 
          (=> (and (>= n 0) (= sum 0) (= i 0)) (inv n sum i))
          (=> (and (inv n sum i) (< (+ (* -1 n) i) 0) (and (= np n) (= sump (+ sum i)) (= ip (+ i 1)))) (inv np sump ip))
          (=> (and (not (< (+ (* -1 n) i) 0)) (inv n sum i)) (>= sum 0))
        )
      )
    )
    (check-sat)
    (get-model)
    