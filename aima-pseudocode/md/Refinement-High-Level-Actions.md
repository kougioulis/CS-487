# REFINEMENT-HIGH-LEVEL-ACTIONS

## AIMA3e

_Refinement (Go(Home, SFO),_  
&emsp;STEPS: _[Drive(Home, SFOLongTermParking ),_  
&emsp;&emsp;&emsp;_Shuttle(SFOLongTermParking, SFO)] )_  
_Refinement (Go(Home, SFO),_  
&emsp;STEPS: _[Taxi(Home, SFO)] )_  

---
_Refinement(Navigate([a, b], [x, y]),_  
&emsp;PRECOND: _a = x ∧ b = y_  
&emsp;STEPS: [ ] )  
_Refinement(Navigate([a, b], [x, y]),_  
&emsp;PRECOND: _Connected ([a, b], [a − 1, b])_  
&emsp;STEPS: _[Left , Navigate([a − 1, b], [x, y])] )_  
_Refinement (Navigate([a, b], [x, y]),_  
&emsp;PRECOND: _Connected ([a, b], [a + 1, b])_  
&emsp;STEPS: _[Right , Navigate([a + 1, b], [x, y])] )_  
...

---
__Figure ??__ Definitions of possible refinements for two high-level actions: going to San Francisco
airport and navigating in the vacuum world. In the latter case, note the recursive nature of the refine-
ments and the use of preconditions.
