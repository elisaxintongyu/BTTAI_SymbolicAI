(define (problem monkey-banana-problem)
  (:domain monkey-banana)
  
  (:objects
    monkey1
    boxD boxB boxC
    banana1
    posA posB posC posD
  )
  
  (:init
    ; Initial positions
    (at monkey1 posD)
    (box-at boxD posA)
    (box-at boxB posB)
    (box-at boxC posC)
    (banana-at banana1 posB)
    
    ; Initial height states
    (on-ground monkey1)
    (box-on-ground boxD)
    (box-on-ground boxB)
    (box-on-ground boxC)
    (banana-on-box banana1 boxB)
    
    ; Adjacency relationships (positions are arranged left to right: A-B-C-D)
    (adjacent posA posB)
    (adjacent posB posA)
    (adjacent posB posC)
    (adjacent posC posB)
    (adjacent posC posD)
    (adjacent posD posC)
  )
  
  (:goal (has-banana monkey1 banana1))
)
