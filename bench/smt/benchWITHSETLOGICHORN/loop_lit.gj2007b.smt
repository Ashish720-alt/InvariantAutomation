(set-logic HORN)

    (declare-fun |inv| (Int Int Int) Bool)
    (assert 
      (forall ((x Int) (m Int) (n Int) (xp Int) (mp Int) (np Int)) 
        (and 
          (=> (and (= x 0) (= m 0) (> n 0) (> n 0)) (inv x m n))
          (=> (and (inv x m n) (< (+ x (* -1 n)) 0) (and (= xp (+ x 1)) (= mp m) (= np n))) (inv xp mp np))
(=> (and (inv x m n) (< (+ x (* -1 n)) 0) (and (= xp (+ x 1)) (= mp x) (= np n))) (inv xp mp np))
          (=> (and (not (< (+ x (* -1 n)) 0)) (inv x m n)) (and (>= m 0) (< (+ m (* -1 n)) 0)))
        )
      )
    )
    (check-sat)
    (get-model)
    