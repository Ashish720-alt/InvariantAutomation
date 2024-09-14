(set-logic HORN)

    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (k Int) (ip Int) (jp Int) (kp Int)) 
        (and 
          (=> (and (= i 1) (= j 1) (>= k 0) (<= k 1)) (inv i j k))
          (=> (and (inv i j k) (< i 1000000) (and (= ip (+ i 1)) (= jp (+ j k)) (= kp (+ k -1)))) (inv ip jp kp))
          (=> (and (not (< i 1000000)) (inv i j k)) (and (<= (+ i k) 2) (>= i 1) (<= (+ (* -1 i) (* -1 k)) -1)))
        )
      )
    )
    (check-sat)
    (get-model)
    