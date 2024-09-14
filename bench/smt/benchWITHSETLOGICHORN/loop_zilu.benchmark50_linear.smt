(set-logic HORN)

    (declare-fun |inv| (Int Int) Bool)
    (assert 
      (forall ((xa Int) (ya Int) (xap Int) (yap Int)) 
        (and 
          (=> (> (+ xa ya) 0) (inv xa ya))
          (=> (and (inv xa ya) (> xa 0) (and (= xap (+ xa -1)) (= yap (+ ya 1)))) (inv xap yap))
          (=> (and (not (> xa 0)) (inv xa ya)) (>= ya 0))
        )
      )
    )
    (check-sat)
    (get-model)
    