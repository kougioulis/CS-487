# GRAPHPLAN

## AIMA3e
__function__ GRAPHPLAN(_problem_) __returns__ solution or failure  

&emsp;_graph_ &larr; INITIAL\-PLANNING\-GRAPH(_problem_)  
&emsp;_goals_ &larr; CONJUNCTS(_problem_.GOAL)  
&emsp;_nogoods_ &larr; an empty hash table  
&emsp;__for__ _tl_ = 0 __to__ &infin; __do__  
&emsp;&emsp;&emsp;__if__ _goals_ all non\-mutex in _S_<sub>t</sub> of _graph_ __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;_solution_ &larr; EXTRACT\-SOLUTION(_graph_, _goals_, NUMLEVELS(_graph_), _nogoods_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ solution &ne; _failure_ __then return__ _solution_  
&emsp;&emsp;&emsp;__if__ _graph_ __and__ _nogoods_ have both leveled off __then return__ _failure_  
&emsp;&emsp;&emsp;_graph_ &larr; EXPAND\-GRAPH(_graph_, _problem_)

---
__Figure ??__ The GRAPHPLAN algorithm. GRAPHPLAN calls EXPAND-GRAPH to add a level until either a solution is found by EXTRACT-SOLUTION, or no solution is possible.
