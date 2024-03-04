# APPEND

## AIMA3e
__procedure__ APPEND(_ax_, _y_, _az_, _continuation_)  

&emsp;_trail_ &larr; GLOBAL-TRAIL-POINTER()  
&emsp;__if__ _ax_ = \[ \] and UNIFY(_y_, _az_) __then__ CALL(_continuation_)  
&emsp;RESET-TRAIL(_trail_)  
&emsp;_a_, _x_, _z_ &larr; NEW-VARIABLE(), NEW-VARIABLE(), NEW-VARIABLE()  
&emsp;__if__ UNIFY(_ax_, \[_a_ | _x_\]) and UNIFY(_az_, \[_a_ | _z_\]) __then__ APPEND(_x_, _y_, _z_, _continuation_)  

---
__Figure__ ?? Pseudocode representing the result of compiling the Append predicate. The function NEW-VARIABLE returns a new variable, distinct from all other variables used so far. The procedure CALL(_continuation_) continues execution with the specified continuation.
