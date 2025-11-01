(define (domain monkey-banana)
  (:requirements :strips)
  
  (:predicates
    ; Location predicates
    (at ?monkey ?location)
    (box-at ?box ?location)
    (banana-at ?banana ?location)
    
    ; Height/level predicates
    (on-ground ?monkey)
    (on-box ?monkey ?box)
    (box-on-ground ?box)
    (banana-on-ground ?banana)
    (banana-on-box ?banana ?box)
    
    ; Adjacency for movement
    (adjacent ?loc1 ?loc2)
    
    ; Goal state
    (has-banana ?monkey ?banana)
  )
  
  (:action move
    :parameters (?monkey ?from ?to)
    :precondition (and (at ?monkey ?from) (adjacent ?from ?to) (on-ground ?monkey))
    :effect (and (at ?monkey ?to) (not (at ?monkey ?from)))
  )
  
  (:action climb_on
    :parameters (?monkey ?box ?location)
    :precondition (and (at ?monkey ?location) (box-at ?box ?location) (on-ground ?monkey))
    :effect (and (on-box ?monkey ?box) (not (on-ground ?monkey)))
  )
  
  (:action climb_off
    :parameters (?monkey ?box ?location)
    :precondition (and (on-box ?monkey ?box) (box-at ?box ?location))
    :effect (and (on-ground ?monkey) (not (on-box ?monkey ?box)))
  )
  
  (:action push_box
    :parameters (?monkey ?box ?from ?to)
    :precondition (and (at ?monkey ?from) (box-at ?box ?from) (adjacent ?from ?to) (on-ground ?monkey))
    :effect (and (box-at ?box ?to) (not (box-at ?box ?from)))
  )
  
  (:action grab_banana_from_ground
    :parameters (?monkey ?banana ?location)
    :precondition (and 
      (at ?monkey ?location) 
      (banana-at ?banana ?location)
      (on-ground ?monkey)
      (banana-on-ground ?banana)
    )
    :effect (has-banana ?monkey ?banana)
  )
  
  (:action grab_banana_from_box
    :parameters (?monkey ?banana ?box ?location)
    :precondition (and 
      (at ?monkey ?location) 
      (banana-at ?banana ?location)
      (on-ground ?monkey)
      (banana-on-box ?banana ?box)
      (box-at ?box ?location)
    )
    :effect (has-banana ?monkey ?banana)
  )
)
