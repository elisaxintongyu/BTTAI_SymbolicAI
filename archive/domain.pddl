(define (domain MONKEY)
  (:requirements :strips :typing)
  
  (:types
    box fruit mammal location
  )
  
  (:predicates
    ;; Spatial relationships
    (LEFT-OF ?x ?y)
    (RIGHT-OF ?x ?y)
    (CLEAR-PATH-BETWEEN ?x ?y)
    
    ;; Object states
    (CLEAR-ON-TOP ?box - box)
    (ON-FLOOR ?obj)
    (ON-BOX ?obj ?box - box)
    (HAND-EMPTY ?m - mammal)
    (HAS ?m - mammal ?f - fruit)
    
    ;; Action enabling predicates
    (CAN-GRAB ?m - mammal ?f - fruit)
    (CAN-CLIMB ?m - mammal ?b - box)
    (AT ?m - mammal ?loc - location)
    (AT-OBJ ?obj ?loc - location)
  )
  
  (:action walk-to
    :parameters (?m - mammal ?loc - location)
    :precondition (and
      (ON-FLOOR ?m)
      (CLEAR-PATH-BETWEEN ?m ?loc)
    )
    :effect (and
      (AT ?m ?loc)
    )
  )
  
  (:action walk-to-object
    :parameters (?m - mammal ?obj ?loc - location)
    :precondition (and
      (ON-FLOOR ?m)
      (CLEAR-PATH-BETWEEN ?m ?obj)
      (AT-OBJ ?obj ?loc)
    )
    :effect (and
      (AT ?m ?loc)
    )
  )
  
  (:action push-box
    :parameters (?m - mammal ?b - box ?from - location ?to - location)
    :precondition (and
      (ON-FLOOR ?m)
      (ON-FLOOR ?b)
      (AT ?m ?from)
      (AT-OBJ ?b ?from)
      (CLEAR-PATH-BETWEEN ?from ?to)
      (HAND-EMPTY ?m)
    )
    :effect (and
      (not (AT ?m ?from))
      (not (AT-OBJ ?b ?from))
      (AT ?m ?to)
      (AT-OBJ ?b ?to)
    )
  )
  
  (:action climb-on-box
    :parameters (?m - mammal ?b - box ?loc - location)
    :precondition (and
      (ON-FLOOR ?m)
      (ON-FLOOR ?b)
      (AT ?m ?loc)
      (AT-OBJ ?b ?loc)
      (CLEAR-ON-TOP ?b)
      (HAND-EMPTY ?m)
    )
    :effect (and
      (not (ON-FLOOR ?m))
      (ON-BOX ?m ?b)
      (not (CLEAR-ON-TOP ?b))
    )
  )
  
  (:action climb-down
    :parameters (?m - mammal ?b - box)
    :precondition (and
      (ON-BOX ?m ?b)
    )
    :effect (and
      (not (ON-BOX ?m ?b))
      (ON-FLOOR ?m)
      (CLEAR-ON-TOP ?b)
    )
  )
  
  (:action enable-grab
    :parameters (?m - mammal ?f - fruit ?loc - location)
    :precondition (and
      (AT ?m ?loc)
      (AT-OBJ ?f ?loc)
      (ON-FLOOR ?f)
    )
    :effect (CAN-GRAB ?m ?f)
  )
  
  (:action enable-grab-from-box
    :parameters (?m - mammal ?f - fruit ?b - box ?loc - location)
    :precondition (and
      (ON-BOX ?m ?b)
      (AT-OBJ ?b ?loc)
      (AT-OBJ ?f ?loc)
      (ON-FLOOR ?f)
    )
    :effect (CAN-GRAB ?m ?f)
  )
  
  (:action enable-grab-box-top
    :parameters (?m - mammal ?f - fruit ?b - box)
    :precondition (and
      (ON-BOX ?m ?b)
      (ON-BOX ?f ?b)
    )
    :effect (CAN-GRAB ?m ?f)
  )
  
  (:action grab
    :parameters (?m - mammal ?f - fruit)
    :precondition (and
      (CAN-GRAB ?m ?f)
      (HAND-EMPTY ?m)
    )
    :effect (and
      (not (HAND-EMPTY ?m))
      (HAS ?m ?f)
      (not (CAN-GRAB ?m ?f))
    )
  )
)
