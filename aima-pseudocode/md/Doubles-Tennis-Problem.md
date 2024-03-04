# DOUBLES-TENNIS-PROBLEM

## AIMA3e
_Actors(A, B)_  
_Init(At (A, LeftBaseline) ∧ At(B, RightNet ) ∧_  
&emsp;_Approaching (Ball, RightBaseline)) ∧ Partner (A, B) ∧ Partner (B, A)_  
_Goal (Returned (Ball) ∧ (At(a, RightNet ) ∨ At (a, LeftNet ))_  
_Action(Hit(actor , Ball),_  
&emsp;PRECOND: _Approaching (Ball, loc) ∧ At(actor , loc)_  
&emsp;EFFECT: _Returned (Ball))_  
_Action(Go(actor , to),_  
&emsp;PRECOND: _At (actor , loc) ∧ to &ne; loc,_  
&emsp;EFFECT: _At (actor , to) ∧ ¬ At (actor , loc))_  

---
__Figure ??__ The doubles tennis problem. Two actors _A_ and _B_ are playing together and can be in
one of four locations: _LeftBaseline_, _RightBaseline_, _LeftNet_ , and _RightNet_ . The ball can be returned
only if a player is in the right place. Note that each action must include the actor as an argument.
