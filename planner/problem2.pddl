(define (problem monkey-banana-problem)
  (:domain monkey-banana)

  (:objects
    monkey
    banana
    box1 box2
    L1 L2 L3 L4  ; abstract locations
  )

  (:init
    ;; Monkey starts at L1
    (at monkey L1)
    (on-ground monkey)

    ;; Boxes
    (box-at box1 L2)
    (box-on-ground box1)

    (box-at box2 L3)
    (box-on-ground box2)

    ;; Banana
    (banana-at banana L4)
    (banana-on-ground banana)

    ;; Relative movement structure
    ;; Monkey must traverse L1 → L2 → L3 → L4
    (adjacent L1 L2)
    (adjacent L2 L1)

    (adjacent L2 L3)
    (adjacent L3 L2)

    (adjacent L3 L4)
    (adjacent L4 L3)
  )

  (:goal
    (has-banana monkey banana)
  )
)
