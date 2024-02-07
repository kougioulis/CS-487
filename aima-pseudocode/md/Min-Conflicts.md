# MIN-CONFLICTS

## AIMA3e
__function__ MIN-CONFLICTS(_csp_, _max\_steps_) __returns__ a solution of failure  
&emsp;__inputs__: _csp_, a constraint satisfaction problem  
&emsp;&emsp;&emsp;&emsp;_max\_steps_, the number of steps allowed before giving up  

&emsp;_current_ &larr; an initial complete assignment for _csp_  
&emsp;__for__ _i_ = 1 to _max\_steps_ __do__  
&emsp;&emsp;&emsp;__if__ _current_ is a solution for _csp_ __then return__ _current_  
&emsp;&emsp;&emsp;_var_ &larr; a randomly chosen conflicted variable from _csp_.VARIABLES  
&emsp;&emsp;&emsp;_value_ &larr; the value _v_ for _var_ that minimizes CONFLICTS(_var_, _v_, _current_, _csp_)  
&emsp;&emsp;&emsp;set _var_ = _value_ in _current_  
&emsp;__return__ _failure_  

---
__Figure__ ?? The MIN-CONFLICTS algorithm for solving CSPs by local search. The initial state may be chosen randomly or by a greedy assignment process that chooses a minimal\-conflict value for each variable in turn. The CONFLICTS function counts the number of constraints violated by a particular value, given the rest of the current assignment.
