# BACKTRACKING-SEARCH

## AIMA3e
__function__ BACKTRACKING-SEARCH(_csp_) __returns__ a solution, or failure  
&emsp;__return__ BACKTRACK(\{\}, _csp_)  

__function__ BACKTRACK(_assignment_, _csp_) __returns__ a solution, or failure  
&emsp;__if__ _assignment_ is complete __then return__ _assignment_  
&emsp;_var_ &larr; SELECT\-UNASSIGNED-VARIABLE(_csp_)  
&emsp;__for each__ _value_ __in__ ORDER\-DOMAIN\-VALUES(_var_, _assignment_, _csp_) __do__  
&emsp;&emsp;&emsp;__if__ _value_ is consistent with _assignment_ __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;add \{_var_ = _value_\} to _assignment_  
&emsp;&emsp;&emsp;&emsp;&emsp;_inferences_ &larr; INFERENCE(_csp_, _var_, _value_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _inferences_ &ne; _failure_ __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;add _inferences_ to _assignment_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_result_ &larr; BACKTRACK(_assignment_, _csp_)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _result_ &ne; _failure_ __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__return__ _result_  
&emsp;&emsp;&emsp;&emsp;&emsp;remove \{_var_ = _value_\} and _inferences_ from _assignment_  
&emsp;__return__ _failure_

---
__Figure__ ?? A simple backtracking algorithm for constraint satisfaction problems. The algorithm is modeled on the recursive depth\-first search of Chapter ??. By varying the functions SELECT\-UNASSIGNED\-VARIABLE and ORDER\-DOMAIN\-VALUES, we can implement the general\-purpose heuristics discussed in the text. The function INFERENCE can optionally be used to impose arc\-,path\-, or _k_\-consistency, as desired. If a value choice leads to failure (noticed either by INFERENCE or by BACKTRACK), then value assignments (including those made by INFERENCE) are removed from the current assignment and a new value is tried.
