(define (problem monkey-banana-generated)
  (:domain monkey-banana)

  (:objects
    monkey
    banana
    box1
    box2
    boxa
    boxb
    l1
    l2
    l3
    l4
  )

  (:init
    (adjacent l1 l2)
    (adjacent l2 l1)
    (adjacent l2 l3)
    (adjacent l3 l2)
    (adjacent l3 l4)
    (adjacent l4 l3)
    (at monkey l1)
    (banana-at banana l4)
    (banana-on-ground banana)
    (box-at box1 l2)
    (box-at box2 l3)
    (box-at boxa l2)
    (box-at boxb l3)
    (box-on-ground box1)
    (box-on-ground box2)
    (box-on-ground boxa)
    (box-on-ground boxb)
    (on-ground monkey)
  )

  (:goal
    (has-banana monkey banana)
  )
)
