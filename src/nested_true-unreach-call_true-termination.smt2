set-info :original "/tmp/sea-LY6fbf/nested_true-unreach-call_true-termination.pp.ms.o.bc")
(set-info :authors "SeaHorn v.0.1.0-rc3")
(declare-rel verifier.error (Bool Bool Bool ))
(declare-rel main@entry (Int ))
(declare-rel main@.lr.ph.us (Int Int Int Int ))
(declare-rel main@_bb (Int Int Int Int Int ))
(declare-rel main@precall.split ())
(declare-var main@%_14_0 Bool )
(declare-var main@%_13_0 Bool )
(declare-var main@%.1.i1.us.lcssa.lcssa_1 Int )
(declare-var main@%_11_0 Bool )
(declare-var main@%.1.i1.us.lcssa_1 Int )
(declare-var main@%.lcssa_1 Int )
(declare-var main@%.04.i2.us_2 Int )
(declare-var main@%.06.i3.us_2 Int )
(declare-var main@%_0_0 Int )
(declare-var @__VERIFIER_nondet_int_0 Int )
(declare-var main@%_2_0 Int )
(declare-var main@%_4_0 Bool )
(declare-var main@%_5_0 Bool )
(declare-var main@%or.cond.i_0 Bool )
(declare-var main@%_6_0 Bool )
(declare-var main@%_7_0 Bool )
(declare-var main@%or.cond_0 Bool )
(declare-var main@entry_0 Bool )
(declare-var main@%_1_0 Int )
(declare-var main@%_3_0 Int )
(declare-var main@.lr.ph5.split.us_0 Bool )
(declare-var main@.lr.ph.us_0 Bool )
(declare-var main@%.05.i4.us_0 Int )
(declare-var main@%.06.i3.us_0 Int )
(declare-var main@%.05.i4.us_1 Int )
(declare-var main@%.06.i3.us_1 Int )
(declare-var main@precall_0 Bool )
(declare-var main@%.05.i.lcssa_0 Bool )
(declare-var main@%.05.i.lcssa_1 Bool )
(declare-var main@precall.split_0 Bool )
(declare-var main@_bb_0 Bool )
(declare-var main@%.04.i2.us_0 Int )
(declare-var main@%.1.i1.us_0 Int )
(declare-var main@%.04.i2.us_1 Int )
(declare-var main@%.1.i1.us_1 Int )
(declare-var main@%_9_0 Int )
(declare-var main@%_10_0 Int )
(declare-var main@._crit_edge.us_0 Bool )
(declare-var main@%.1.i1.us.lcssa_0 Int )
(declare-var main@%.lcssa_0 Int )
(declare-var main@%_12_0 Int )
(declare-var main@_bb_1 Bool )
(declare-var main@%.1.i1.us_2 Int )
(declare-var main@precall.loopexit_0 Bool )
(declare-var main@%.1.i1.us.lcssa.lcssa_0 Int )
(declare-var main@%phitmp_0 Bool )
(rule (verifier.error false false false))
(rule (verifier.error false true true))
(rule (verifier.error true false true))
(rule (verifier.error true true true))

; precondition
(rule (main@entry @__VERIFIER_nondet_int_0))
; precondition
; This is a precondition because main@entry is used on the LHS.
(rule (=> (and (main@entry @__VERIFIER_nondet_int_0))
          (main@.lr.ph.us 0 4 0 4)))
; This is a short-cut rule in the case that the loop is not executed.
(rule (=> (and (main@entry @__VERIFIER_nondet_int_0)
               (not (and (> 4 0) (> 4 0))))
          main@precall.split))
; transition
(rule (=> (and (main@.lr.ph.us main@%.06.i3.us_0
                               main@%_1_0
                               main@%.05.i4.us_0
                               main@%_3_0))
          (main@_bb main@%.06.i3.us_0
                    main@%_1_0
                    main@%.05.i4.us_0
                    0
                    main@%_3_0)))
; transition 
; This is a transition rule because both its input and output relation are not main@entry or main@precall.split (which is query at last).
; Let's say main@_bb is I1 and its first parameters are x1,x2,..,x5, and main@.lr.ph.us is I2.
; Then this rule is
;    I1(x1,x2,x3,x4,x5) /\ ~(x4+1 < x5) /\ (x1+1 < x2) => I2(x1+1,x2,x3+1,x5)
(rule (=> (and (main@_bb main@%.06.i3.us_0
                         main@%_1_0
                         main@%.1.i1.us_0
                         main@%.04.i2.us_0
                         main@%_3_0)
               (not (< (+ main@%.04.i2.us_0 1) main@%_3_0))
               (< (+ main@%.06.i3.us_0 1) main@%_1_0))
          (main@.lr.ph.us (+ main@%.06.i3.us_0 1)
                          main@%_1_0
                          (+ main@%.1.i1.us_0 1)
                          main@%_3_0)))
; transition
(rule (=> (and (main@_bb main@%.06.i3.us_0
                         main@%_1_0
                         main@%.1.i1.us_0
                         main@%.04.i2.us_0
                         main@%_3_0)
               (< (+ main@%.04.i2.us_0 1) main@%_3_0))
          (main@_bb main@%.06.i3.us_0
                    main@%_1_0
                    (+ main@%.1.i1.us_0 1)
                    (+ main@%.04.i2.us_0 1)
                    main@%_3_0)))
; postcondition
; Intuitively, this is a postcondition because the RHS is main@precall.split,
; which will be query at last.
(rule (=> (and (main@_bb main@%.06.i3.us_0
                         main@%_1_0
                         main@%.1.i1.us_0
                         main@%.04.i2.us_0
                         main@%_3_0)
               (not (< (+ main@%.04.i2.us_0 1) main@%_3_0))
               (not (< (+ main@%.06.i3.us_0 1) main@%_1_0))
               (< main@%.1.i1.us_0 15))
          main@precall.split))
; asking if postcondition can be satisfied or not
(query main@precall.split)

