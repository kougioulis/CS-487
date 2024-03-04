# ONLINE-DFS-AGENT

## AIMA3e
__function__ ONLINE-DFS-AGENT(_s'_) __returns__ an action  
&emsp;__inputs__: _s'_, a percept that identifies the current state  
&emsp;__persistent__: _result_, a table indexed by state and action, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_untried_, a table that lists, for each state, the actions not yet tried  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_unbacktracked_, a table that lists, for each state, the backtracks not yet tried  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_s_, _a_, the previous state and action, initially null  

&emsp;__if__ GOAL\-TEST(_s'_) __then return__ _stop_  
&emsp;__if__ _s'_ is a new state (not in _untried_) __then__ _untried_\[_s'_\] &larr; ACTIONS(_s'_)  
&emsp;__if__ _s_ is not null and _s'_ != _result_\[_s_, _a_\] __then__  
&emsp;&emsp;&emsp;_result_\[_s_, _a_\] &larr; _s'_  
&emsp;&emsp;&emsp;add _s_ to front of _unbacktracked_\[_s'_\]  
&emsp;__if__ _untried_\[_s'_\] is empty __then__  
&emsp;&emsp;&emsp;__if__ _unbacktracked_\[_s'_\] is empty __then return__ _stop_  
&emsp;&emsp;&emsp;__else__ _a_ &larr; an action _b_ such that _result_\[_s'_, _b_\] = POP(_unbacktracked_\[_s'_\])  
&emsp;__else__ _a_ &larr; POP(_untried_\[_s'_\])  
&emsp;_s_ &larr; _s'_  
&emsp;__return__ _a_  

---
__Figure__ ?? An online search agent that uses depth-first exploration. The agent is applicable only in state spaces in which every action can be "undone" by some other action.
