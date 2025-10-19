(define (domain MONKEY)
  (:requirements :strips :typing)
  (:types mammal fruit box)

  (:predicates
    (CLEAR-PATH-BETWEEN ?x ?y)
    (CAN-GRAB ?x ?y)
    (HAS ?x ?y)
    (CLEAR-ON-TOP ?x)
    (LEFT-OF ?x ?y)
    (RIGHT-OF ?x ?y)
    (ON-FLOOR ?x)
    (HAND-EMPTY ?x)
  )

  ;; Mammal can walk somewhere if the path is clear
  (:action walk-to
    :parameters (?m - mammal ?target - (either box fruit))
    :precondition (CLEAR-PATH-BETWEEN ?m ?target)
    :effect (CAN-GRAB ?m ?target)
  )

  ;; Mammal can grab the fruit if it can reach it
  (:action grab
    :parameters (?m - mammal ?f - fruit)
    :precondition (and
      (CAN-GRAB ?m ?f)
      (HAND-EMPTY ?m)
    )
    :effect (and
      (HAS ?m ?f)
      (not (HAND-EMPTY ?m))
    )
  )

  ;; Mammal can drop what it’s holding
  (:action drop
    :parameters (?m - mammal ?f - fruit)
    :precondition (HAS ?m ?f)
    :effect (and
      (not (HAS ?m ?f))
      (HAND-EMPTY ?m)
    )
  )
)
