(set-logic HORN)

    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (k Int) (ip Int) (jp Int) (kp Int)) 
        (and 
          (=> (and (< (+ i (* -1 j)) 0) (> k 0)) (inv i j k))
          (=> (and (inv i j k) (< (+ i (* -1 j)) 0) (and (= ip (+ i 1)) (= jp j) (= kp (+ k 1)))) (inv ip jp kp))
          (=> (and (not (< (+ i (* -1 j)) 0)) (inv i j k)) (> (+ i (+ (* -1 j) k)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    