
    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((n Int) (i Int) (l Int) (np Int) (ip Int) (lp Int)) 
        (and 
          (=> (and (> l 0) (= (+ i (* -1 l)) 0)) (inv n i l))
          (=> (and (inv n i l) (< (+ (* -1 n) i) 0)   (and (= np n) (= ip (+ i 1)) (= lp l))) (inv np ip lp))
          (=> (and (not (< (+ (* -1 n) i) 0)) (inv n i l)) (>= l 1))
        )
      )
    )
    (check-sat)
    (get-model)
    