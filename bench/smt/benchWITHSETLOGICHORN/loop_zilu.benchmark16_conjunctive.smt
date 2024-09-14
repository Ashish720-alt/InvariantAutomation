(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((i Int) (k Int) (ip Int) (kp Int)) 
        (and 
          (=> (and (<= (* -1 k) 0) (<= k 1) (= i 1)) (inv i k))
          (=> (and (inv i k) (and (= ip (+ i 1)) (= kp (+ k -1)))) (inv ip kp))
          (=> (inv i k) (and (<= (+ (* -1 i) (* -1 k)) -1) (<= (+ i k) 2) (>= i 1)))
        )
      )
    )
    (check-sat)
    (get-model)
    