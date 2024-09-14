
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((i Int) (k Int) (n Int) (ip Int) (kp Int) (np Int)) 
        (and 
          (=> (and (= i 0) (= k 0) (> n 0)) (inv i k n))
          (=> (and (inv i k n) (< (+ i (* -1 n)) 0)   (and (= ip (+ i 1)) (= kp (+ k 1)) (= np n))) (inv ip kp np))
          (=> (and (not (< (+ i (* -1 n)) 0)) (inv i k n)) (and (= (+ k (* -1 n)) 0) (= (+ i (* -1 k)) 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    