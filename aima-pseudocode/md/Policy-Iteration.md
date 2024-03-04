# POLICY-ITERATION

## AIMA3e
__function__ POLICY-ITERATION(_mdp_) __returns__ a policy  
&emsp;__inputs__: _mdp_, an MDP with states _S_, actions _A_(_s_), transition model _P_(_s&prime;_ &vert; _s_, _a_)  
&emsp;__local variables__: _U_, a vector of utilities for states in _S_, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_&pi;_, a policy vector indexed by state, initially random  

&emsp;__repeat__  
&emsp;&emsp;&emsp;_U_ &larr; POLICY\-EVALUATION(_&pi;_, _U_, _mdp_)  
&emsp;&emsp;&emsp;_unchanged?_ &larr; true  
&emsp;&emsp;&emsp;__for each__ state _s_ __in__ _S_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ max<sub>_a_ &isin; _A_(_s_)</sub> &Sigma;<sub>_s&prime;_</sub> _P_(_s&prime;_ &vert; _s_, _a_) _U_\[_s&prime;_\] &gt; &Sigma;<sub>_s&prime;_</sub> _P_(_s&prime;_ &vert; _s_, _&pi;_\[_s_\]) _U_\[_s&prime;_\] __then do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_&pi;_\[_s_\] &larr; argmax<sub>_a_ &isin; _A_(_s_)</sub> &Sigma;<sub>_s&prime;_</sub> _P_(_s&prime;_ &vert; _s_, _a_) _U_\[_s&prime;_\]  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_unchanged?_ &larr; false  
&emsp;__until__ _unchanged?_  
&emsp;__return__ _&pi;_  

---
__Figure ??__ The policy iteration algorithm for calculating an optimal policy.

---
  
## AIMA4e  
__function__ POLICY-ITERATION(_mdp_) __returns__ a policy  
&emsp;__inputs__: _mdp_, an MDP with states _S_, actions _A_(_s_), transition model _P_(_s&prime;_ &vert; _s_, _a_)  
&emsp;__local variables__: _U_, a vector of utilities for states in _S_, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_&pi;_, a policy vector indexed by state, initially random  
  
&emsp;__repeat__  
&emsp;&emsp;&emsp;_U_ &larr; POLICY\-EVALUATION(_&pi;_, _U_, _mdp_)  
&emsp;&emsp;&emsp;_unchanged?_ &larr; true  
&emsp;&emsp;&emsp;__for each__ state _s_ __in__ _S_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_a <sup> &#x2a; </sup>_ &larr; argmax<sub>_a_ &isin; _A_(_s_)</sub> Q-VALUE(_mdp_,_s_,_a_,_U_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ Q-VALUE(_mdp_,_s_,_a<sup>&#x2a;</sup>_,_U_) &gt; Q-VALUE(_mdp_,_s_,_&pi;_\[_s_\],_U_) __then do__    
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_&pi;_\[_s_\] &larr; _a<sup>&#x2a;</sup>_ ; _unchanged?_ &larr; false  
&emsp;__until__ _unchanged?_  
&emsp;__return__ _&pi;_  

---
__Figure ??__ The policy iteration algorithm for calculating an optimal policy.  
