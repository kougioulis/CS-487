# ANGELIC-SEARCH

## AIMA3e
__function__ ANGELIC-SEARCH(_problem_, _hierarchy_, _initialPlan_) __returns__ solution or _fail_  
&emsp;_frontier_ &larr; a FIFO queue with _initialPlan_ as the only element  
&emsp;__loop do__  
&emsp;&emsp;&emsp;__if__ EMPTY?(_frontier_) __then return__ _fail_  
&emsp;&emsp;&emsp;_plan_ &larr; POP(_frontier_) /\* chooses the shallowest node in _frontier_ \*/  
&emsp;&emsp;&emsp;__if__ REACH<sup>+</sup>(_problem_.INITIAL\-STATE, _plan_) intersects _problem_.GOAL __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _plan_ is primitive __then return__ _plan_ /\* REACH<sup>+</sup> is exact for primitive plans \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;_guaranteed_ &larr; REACH<sup>&minus;</sup>(_problem_.INITIAL\-STATE, _plan_) &cap; _problem_.GOAL  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _guaranteed_ &ne; \{\} and MAKING\-PROGRESS(_plan_, _initialPlan_) __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_finalState_ &larr; any element of _guaranteed_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__return__ DECOMPOSE(_hierarchy_, _problem_.INITIAL\-STATE, _plan_, _finalState_)  
&emsp;&emsp;&emsp;&emsp;&emsp;_hla_ &larr; some HLA in _plan_  
&emsp;&emsp;&emsp;&emsp;&emsp;_prefix_,_suffix_ &larr; the action subsequences before and after _hla_ in _plan_  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ _sequence_ __in__ REFINEMENTS(_hla_, _outcome_, _hierarchy_) __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_frontier_ &larr; INSERT(APPEND(_prefix_, _sequence_, _suffix_), _frontier_)  

---
__function__ DECOMPOSE(_hierarchy_, _s<sub>0</sub>_, _plan_, _s<sub>f</sub>_) __returns__ a solution  
&emsp;_solution_ &larr; an empty plan  
&emsp;__while__ _plan_ is not empty __do__  
&emsp;&emsp;&emsp;_action_ &larr; REMOVE\-LAST(_plan_)  
&emsp;&emsp;&emsp;_s<sub>i</sub>_ &larr; a state in REACH<sup>&minus;</sup>(_s<sub>0</sub>_, _plan_) such that _s<sub>f</sub>_ &isin; REACH<sup>&minus;</sup>(_s<sub>i</sub>_, _action_)  
&emsp;&emsp;&emsp;_problem_ &larr; a problem with INITIAL\-STATE = _s<sub>i</sub>_ and GOAL = _s<sub>f</sub>_  
&emsp;&emsp;&emsp;_solution_ &larr; APPEND(ANGELIC-SEARCH(_problem_, _hierarchy_, _action_), _solution_)  
&emsp;&emsp;&emsp;_s<sub>f</sub>_ &larr; _s<sub>i</sub>_  
&emsp;__return__ _solution_  

---
__Figure__ ?? A hierarchical planning algorithm that uses angelic semantics to identify and commit to high\-level plans that work while avoiding high\-level plans that don't. The predicate MAKING\-PROGRESS checks to make sure that we aren't stuck in an infinite regression of refinements. At top level, call ANGELIC-SEARCH with \[_Act_\] as the _initialPlan_.
