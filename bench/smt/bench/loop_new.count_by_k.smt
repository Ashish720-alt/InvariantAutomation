
    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((i Int) (k Int) (ip Int) (kp Int)) 
        (and 
          (=> (and (= i 0) (>= k 0) (<= k 10)) (inv i k))
          (=> (and (inv i k) (< (+ i (* -1000000 k)) 0)   (and (= ip (+ i k)) (= kp k))) (inv ip kp))
          (=> (and (not (< (+ i (* -1000000 k)) 0)) (inv i k)) (= (+ i (* -1000000 k)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    