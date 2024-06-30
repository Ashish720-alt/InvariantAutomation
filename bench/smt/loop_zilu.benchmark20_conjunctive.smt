
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((i Int) (n Int) (sum Int) (ip Int) (np Int) (sump Int)) 
        (and 
          (=> (and (= i 0) (>= n 0) (<= n 100) (= sum 0)) (inv i n sum))
          (=> (and (inv i n sum) (< (+ i (* -1 n)) 0) (and (= ip (+ i 1)) (= np n) (= sump (+ i sum)))) (inv ip np sump))
          (=> (and (not (< (+ i (* -1 n)) 0)) (inv i n sum)) (>= sum 0))
        )
      )
    )
    (check-sat)
    (get-model)
    