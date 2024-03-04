# AC-3

## AIMA3e
__function__ AC-3(_csp_) __returns__ false if an inconsistency is found and true otherwise  
&emsp;__inputs__: _csp_, a binary CSP with components (_X_, _D_, _C_)  
&emsp;__local variables__: _queue_, a queue of arcs, initially all the arcs in _csp_  

&emsp;__while__ _queue_ is not empty __do__  
&emsp;&emsp;&emsp;(_X<sub>i</sub>_, _X<sub>j</sub>_) &larr; REMOVE\-FIRST(_queue_)  
&emsp;&emsp;&emsp;__if__ REVISE(_csp_, _X<sub>i</sub>_, _X<sub>j</sub>_) __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ size of _D<sub>i</sub>_ = 0 __then return__ _false_  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ _X<sub>k</sub>_ in _X<sub>i</sub>_.NEIGHBORS &minus; \{_X<sub>j</sub>_\} __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;add(_X<sub>k</sub>_, _X<sub>i</sub>_) to _queue_  
&emsp;__return__ _true_

---
__function__ REVISE(_csp_, _X<sub>i</sub>_, _X<sub>j</sub>_) __returns__ true iff we revise the domain of _X<sub>i</sub>_  
&emsp;_revised_ &larr; _false_  
&emsp;__for each__ _x_ __in__ _D<sub>i</sub>_ __do__  
&emsp;&emsp;&emsp;__if__ no value _y_ in _D<sub>j</sub>_ allows (_x_, _y_) to satisfy the constraint between  _X<sub>i</sub>_ and _X<sub>j</sub>_ __then__  
&emsp;&emsp;&emsp;&emsp;delete _x_ from _D<sub>i</sub>_  
&emsp;&emsp;&emsp;&emsp;_revised_ &larr; _true_  
&emsp;__return__ _revised_  

---
__Figure__ ?? The arc\-consistency algorithm AC\-3. After applying AC\-3, either every arc is arc\-consistent, or some variable has an empty domain, indicating that the CSP cannot be solved. The name "AC\-3" was used by the algorithm's inventor (Mackworth, 1977) because it's the third version developed in the paper.
