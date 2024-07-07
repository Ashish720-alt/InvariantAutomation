
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((j Int) (k Int) (n Int) (jp Int) (kp Int) (np Int)) 
        (and 
          (=> (and (= (+ j (* -1 n)) 0) (= (+ k (* -1 n)) 0) (> n 0)) (inv j k n))
          (=> (and (inv j k n) (and (> n 0) (> j 0)) (and (= jp (+ j -1)) (= kp (+ k -1)) (= np n))) (inv jp kp np))
          (=> (and (not (and (> n 0) (> j 0))) (inv j k n)) (and (= k 0) (= k 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    