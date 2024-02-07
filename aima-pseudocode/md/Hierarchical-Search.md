# HIERARCHICAL-SEARCH

## AIMA3e
__function__ HIERARCHICAL-SEARCH(_problem_, _hierarchy_) __returns__ a solution, or failure  
&emsp;_frontier_ &larr; a FIFO queue with \[_Act_\] as the only element  
&emsp;__loop do__  
&emsp;&emsp;&emsp;__if__ EMPTY?(_frontier_) __then return__ _failure_  
&emsp;&emsp;&emsp;_plan_ &larr; POP(_frontier_) /\* chooses the shallowest plan in frontier \*/  
&emsp;&emsp;&emsp;_hla_ &larr; the first HLA in _plan_, or _null_ if none  
&emsp;&emsp;&emsp;_prefix_,_suffix_ &larr; the action subsequences before and after _hla_ in plan  
&emsp;&emsp;&emsp;_outcome_ &larr; RESULT(_problem_.INITIAL\-STATE, _prefix_)  
&emsp;&emsp;&emsp;__if__ _hla_ is null __then__ /\* so _plan_ is primitive and _outcome_ is its result \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _outcome_ satisfies _problem_.GOAL __then return__ _plan_  
&emsp;&emsp;&emsp;__else for each__ _sequence_ __in__ REFINEMENTS(_hla_, _outcome_, _hierarchy_)  __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_frontier_ &larr; INSERT(APPEND(_prefix_, _sequence_, _suffix_), _frontier_)  

---
__Figure__ ?? A breadth\-first implementation of hierarchical forward planning search. The initial plan supplied to the algorithm is \[_Act_\]. The REFINEMENTS function returns a set of action sequences, one for each refinement of the HLA whose preconditions are satisfied by the specified state, _outcome_.
