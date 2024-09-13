
    (declare-fun |inv| (Int Int Int Int) Bool)
    (assert 
      (forall ((i Int) (n Int) (k Int) (flag Int) (ip Int) (np Int) (kp Int) (flagp Int)) 
        (and 
          (=> (and (> k 0) (> n 0) (< n 10) (= i 0)) (inv i n k flag))
          (=> (and (inv i n k flag) (< (+ i (* -1 n)) 0) (and (= ip (+ i 1)) (= np n) (= kp (+ k 4000)) (= flagp flag))) (inv ip np kp flagp))
(=> (and (inv i n k flag) (< (+ i (* -1 n)) 0) (and (= ip (+ i 1)) (= np n) (= kp (+ k 2000)) (= flagp flag))) (inv ip np kp flagp))
          (=> (and (not (< (+ i (* -1 n)) 0)) (inv i n k flag)) (> (+ (* -1 n) k) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    