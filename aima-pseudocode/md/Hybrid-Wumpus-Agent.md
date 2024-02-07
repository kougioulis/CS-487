# HYBRID-WUMPUS-AGENT

## AIMA3e
__function__ HYBRID-WUMPUS-AGENT(_percept_) __returns__ an _action_  
&emsp;__inputs__: _percept_, a list, \[_stench_, _breeze_, _glitter_, _bump_, _scream_\]  
&emsp;__persistent__: _KB_, a knowledge base, initially the atemporal "wumpus physics"  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_t_, a counter, initially 0, indicating time  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_plan_, an action sequence, initially empty  

&emsp;TELL(_KB_, MAKE-PERCEPT-SENTENCE(_percept_, _t_))  
&emsp;TELL the _KB_ the temporal "physics" sentences for time _t_  
&emsp;_safe_ &larr; {\[_x_, _y_\] : ASK(_KB_, _OK_<sup>_t_</sup><sub>_x_,_y_</sub>) = _true_}  
&emsp;__if__ ASK(_KB_, _Glitter_<sup>_t_</sup>) = _true_ __then__  
&emsp;&emsp;&emsp;_plan_ &larr; \[_Grab_\] + PLAN-ROUTE(_current_, {\[1,1\]}, _safe_) + \[_Climb_\]  
&emsp;__if__ _plan_ is empty __then__  
&emsp;&emsp;&emsp;_unvisited_ &larr; {\[_x_, _y_\] : ASK(_KB_, _L_<sup>_t'_</sup><sub>_x_,_y_</sub>) = _false_ for all _t'_ &le; _t_}  
&emsp;&emsp;&emsp;_plan_ &larr; PLAN-ROUTE(_current_, _unvisited_ &cap; _safe_, _safe_)  
&emsp;__if__ _plan_ is empty and ASK(_KB_, _HaveArrow_<sup>_t_</sup>) = _true_ __then__  
&emsp;&emsp;&emsp;_possible\_wumpus_ &larr; {\[_x_, _y_\] : ASK(_KB_, &not;_W_<sub>_x_,_y_</sub>) = _false_}  
&emsp;&emsp;&emsp;_plan_ &larr; PLAN-SHOT(_current_, _possible\_wumpus_, _safe_)  
&emsp;__if__ _plan_ is empty __then__ //no choice but to take a risk  
&emsp;&emsp;&emsp;_not\_unsafe_ &larr; {\[_x_, _y_\] : ASK(_KB_, &not;_OK_<sup>_t_</sup><sub>_x_,_y_</sub>) = _false_}  
&emsp;&emsp;&emsp;_plan_ &larr; PLAN-ROUTE(_current_, _unvisited_ &cap; _not\_unsafe_, _safe_)  
&emsp;__if__ _plan_ is empty __then__  
&emsp;&emsp;&emsp;_plan_ &larr; PLAN-ROUTE(_current_, {\[1,1\]}, _safe_) + \[_Climb_\]  
&emsp;_action_ &larr; POP(_plan_)  
&emsp;TELL(_KB_, MAKE-ACTION-SENTENCE(_action_, _t_))  
&emsp;_t_ &larr; _t_ + 1  
&emsp;__return__ _action_

---
__function__ PLAN-ROUTE(_current_, _goals_, _allowed_) __returns__ an action sequence  
&emsp;__inputs__: _current_, the agent's current position  
&emsp;&emsp;&emsp;&emsp;&emsp;_goals_, a set of squares; try to plan a route to one of them  
&emsp;&emsp;&emsp;&emsp;&emsp;_allowed_, a set of squares that can form part of the route  

&emsp;_problem_ &larr; ROUTE-PROBLEM(_current_, _goals_, _allowed_)  
&emsp;__return__ A\*\-GRAPH-SEARCH(_problem_)  

__Figure__ ?? A hybrid agent program for the wumpus world. It uses a propositional knowledge base to infer the state of the world, and a combination of problem-solving search and domain-specific code to decide what actions to take.