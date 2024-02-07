# PASSIVE-TD-AGENT

## AIMA3e
__function__ Passive-TD-Agent(_percept_) __returns__ an action  
&emsp;__inputs__: _percept_, a percept indication the current state _s'_ and reward signal _r'_  
&emsp;__persistent__: _&pi;_, a fixed policy  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_U_, a table of utilities, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_N<sub>s</sub>_, a table of frequencies for states, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_s_, _a_, _r_, the previous state, action, and reward, initially null  

&emsp;__if__ _s'_ is new __then__ _U_[_s'_] &larr; _r'_  
&emsp;__if__ _s_ is not null __then__  
&emsp;&emsp;&emsp;increment _N<sub>s</sub>_[_s_]  
&emsp;&emsp;&emsp;_U_[_s_] &larr; _U_[_s_] + _&alpha;_(_N<sub>s</sub>_[_s_])(r + _&gamma;_ _U_[_s'_] - _U_[_s_])  
&emsp;__if__ _s'_.Terminal? __then__ _s_, _a_, _r_ &larr; null __else__ _s_, _a_, _r_ &larr; _s'_, _&pi;_[_s'_], _r'_  
&emsp;return _a_

---
__Figure ??__ A passive reinforcement learning agent that learns utility estimates using temporal differences. The step-size function &alpha;(_n_) is chosen to ensure convergence, as described in the text.
