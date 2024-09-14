(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((x Int) (n Int) (xp Int) (np Int)) 
        (and 
          (=> (and (= x 0) (> n 0)) (inv x n))
          (=> (and (inv x n) (< (+ x (* -1 n)) 0) (and (= xp (+ x 1)) (= np n))) (inv xp np))
          (=> (and (not (< (+ x (* -1 n)) 0)) (inv x n)) (= (+ x (* -1 n)) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    