# PASSIVE-ADP-AGENT

## AIMA3e
__function__ Passive-ADP-Agent(_percept_) __returns__ and action  
&emsp;__inputs__: _percept_, a percept indication the current state _s'_ and reward signal _r'_  
&emsp;__persistent__: _&pi;_, a fixed policy  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_mdp_, an MDP with model _P_, rewards _R_, discount &gamma;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_U_, a table of utilities, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_N<sub>sa</sub>_, a table of frequencies for state-action pairs, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_N<sub>s'|sa</sub>_, a table of outcome frequencies given state-action pairs, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_s_, _a_, the previous state and action, initially null  
&emsp;__if__ _s'_ is new __then__ _U_[_s'_] &larr; _r'_; _R_[_s'_] &larr; _r'_  
&emsp;__if__ _s_ is not null __then__  
&emsp;&emsp;&emsp;increment _N<sub>sa</sub>_[_s_, _a_] and _N<sub>s'|sa</sub>_[_s'_, _s_, _a_]  
&emsp;&emsp;&emsp;__for each__ _t_ such that _N<sub>s'|sa</sub>_[_t_, _s_, _a_] is nonzero __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_P_(_t_ | _s_, _a_) &larr; _N<sub>s'|sa</sub>_[_t_, _s_, _a_] / _N<sub>sa</sub>_[_s_, _a_]  
&emsp;_U_ &larr; Policy-Evaluation(_&pi;_, _U_, _mdp_)  
&emsp;__if__ _s'_.Terminal? __then__ _s_, _a_ &larr; null __else__ _s_, _a_ &larr; _s'_, _&pi;_[_s'_]  

---
__Figure ??__ A passive reinforcement learning agent based on adaptive dynamic programming. The Policy-Evaluation function solves the fixed-policy Bellman equations, as described on page ??.
