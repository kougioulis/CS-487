# LRTA\*\-AGENT

## AIMA3e
__function__ LRTA\*\-AGENT(_s'_) __returns__ an action  
&emsp;__inputs__: _s'_, a percept that identifies the current state  
&emsp;__persistent__: _result_, a table, indexed by state and action, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_H_, a table of cost estimates indexed by state, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_s_, _a_, the previous state and action, initially null  

&emsp;__if__ GOAL\-TEST(_s'_) __then return__ _stop_  
&emsp;__if__ _s'_ is a new state (not in _H_) __then__ _H_\[_s'_\] &larr; _h_(_s'_)  
&emsp;__if__ _s_ is not null  
&emsp;&emsp;&emsp;_result_\[_s_, _a_\] &larr; _s'_  
&emsp;&emsp;&emsp;_H_\[_s_\] &larr; &emsp;__min__&emsp; LRTA\*\-COST(_s_, _b_, _result_\[_s_, _b_\], _H_)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<sub>_b_ &Element; ACTIONS(_s_)</sub>  
&emsp;_a_ &larr; an action _b_ in ACTIONS(_s'_) that minimizes LRTA\*\-COST(_s'_, _b_, _result_\[_s'_, _b_\], _H_)  
&emsp;_s_ &larr; _s'_  
&emsp;__return__ a

__function__ LRTA\*\-COST(_s_, _a_, _s'_, _H_) __returns__ a cost estimate  
&emsp;__if__ _s'_ is undefined __then return__ _h_(_s_)  
&emsp;__else return__ c(_s_, _a_, _s'_) + _H_\[_s'_\]  

---
__Figure__ ?? LRTA\*\-AGENT selects an action according to the values of neighboring states, which are updated as the agent moves about the state space.
