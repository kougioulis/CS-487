# Q-LEARNING-AGENT

#AIMA3e
__function__ Q-Learning_Agent(_percept_) __returns__ an action  
&emsp;__inputs__: _percept_, a percept indicating the current state _s'_ and reward signal _r'_  
&emsp;__persistent__: _Q_, a table of action values indexed by state and action, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_N<sub>sa</sub>_, a table of frequencies for state-action pairs, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_s_, _a_, _r_, the previous state, action, and reward, initially null  

&emsp;__if__ Terminal?(_s_) then _Q_[_s_, None] &larr; _r'_  
&emsp;__if__ _s_ is not null __then__  
&emsp;&emsp;&emsp;increment _N<sub>sa</sub>_[_s_, _a_]  
&emsp;&emsp;&emsp;_Q_[_s_, _a_] &larr; _Q_[_s_, _a_] + _&alpha;_(_N<sub>sa</sub>_[_s_, _a_])(_r_ + _&gamma;_ max<sub>a'</sub> _Q_[_s'_, _a'_] - _Q_[_s_, _a_])  
&emsp;_s_, _a_, _r_ &larr; _s'_, argmax<sub>a'</sub> _f_(_Q_[_s'_, _a'_], _N<sub>sa</sub>_[_s'_, _a'_]), _r'_  
&emsp;__return__ _a_  

---
__Figure ??__ An exploratory Q-learning agent. It is an active learner that learns the value _Q_(_s_, _a_) of each action in each situation. It uses the same exploration function _f_ as the exploratory ADP agent, but avoids having to learn the transition model because the Q-value of a state can be related directly to those of its neighbors.
