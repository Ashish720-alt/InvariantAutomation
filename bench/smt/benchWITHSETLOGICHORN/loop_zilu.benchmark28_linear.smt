(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((i Int) (j Int) (ip Int) (jp Int)) 
        (and 
          (=> (or (and (< (+ i (* -1 j)) 0) (> (+ i j) 0) (>= j 0)) (and (> (+ i (* -1 j)) 0) (< (+ i j) 0) (>= j 0))) (inv i j))
          (=> (< (+ (* -2 i) j) 0) (=> (and (inv i j) (< (+ i (* -1 j)) 0) (and (= ip (+ (* -1 i) j)) (= jp i))) (inv ip jp)))
(=> (not (or  (< (+ (* -2 i) j) 0))) (=> (>= (+ (* -2 i) j) 0) (=> (and (inv i j) (< (+ i (* -1 j)) 0) (and (= ip i) (= jp (+ (* -1 i) j)))) (inv ip jp))))
          (=> (and (not (< (+ i (* -1 j)) 0)) (inv i j)) (= (+ (* -1 i) j) 0))
        )
      )
    )
    (check-sat)
    (get-model)
    