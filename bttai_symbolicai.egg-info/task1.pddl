(define (problem canvas_0_banana1_monkey1_box2)

(:domain MONKEY)

(:objects
A B - box
banana - fruit
monkey - mammal)

(:init (CLEAR-ON-TOP A) (CLEAR-ON-TOP B) (LEFT-OF B A) (RIGHT-OF A B) (LEFT-OF monkey A) (RIGHT-OF monkey B) (CLEAR-PATH-BETWEEN monkey B) (CLEAR-PATH-BETWEEN monkey banana)
(ON-FLOOR banana) (ON-FLOOR monkey) (ON-FLOOR A) (ON-FLOOR B) (HAND-EMPTY monkey))

(:goal (HAS monkey banana))

)

