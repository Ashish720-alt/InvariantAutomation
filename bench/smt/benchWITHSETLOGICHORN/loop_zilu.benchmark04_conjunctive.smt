(set-logic HORN)

    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((k Int) (j Int) (n Int) (kp Int) (jp Int) (np Int)) 
        (and 
          (=> (and (>= n 1) (>= (+ k (* -1 n)) 0) (= j 0)) (inv k j n))
          (=> (and (inv k j n) (<= (+ j (* -1 n)) -1) (and (= kp (+ k -1)) (= jp (+ j 1)) (= np n))) (inv kp jp np))
          (=> (and (not (<= (+ j (* -1 n)) -1)) (inv k j n)) (>= k 0))
        )
      )
    )
    (check-sat)
    (get-model)
    