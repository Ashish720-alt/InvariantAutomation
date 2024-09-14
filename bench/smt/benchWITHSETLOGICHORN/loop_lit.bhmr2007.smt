(set-logic HORN)

    (declare-fun |inv| (Int Int Int Int) Bool)
    (assert 
      (forall ((a Int) (b Int) (i Int) (n Int) (ap Int) (bp Int) (ip Int) (np Int)) 
        (and 
          (=> (and (= a 0) (= b 0) (= i 0) (>= n 0) (<= n 1000000)) (inv a b i n))
          (=> (and (inv a b i n) (< (+ i (* -1 n)) 0) (and (= ap (+ a 1)) (= bp (+ b 2)) (= ip (+ i 1)) (= np n))) (inv ap bp ip np))
(=> (and (inv a b i n) (< (+ i (* -1 n)) 0) (and (= ap (+ a 2)) (= bp (+ b 1)) (= ip (+ i 1)) (= np n))) (inv ap bp ip np))
          (=> (and (not (< (+ i (* -1 n)) 0)) (inv a b i n)) (or (= (+ a (+ b (* -3 n))) 0) (<= (+ i (* -1 n)) 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    