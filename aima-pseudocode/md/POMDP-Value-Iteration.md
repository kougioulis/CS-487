# POMDP-VALUE-ITERATION

## AIMA3e
__function__ POMDP-VALUE-ITERATION(_pomdp_, _&epsi;_) __returns__ a utility function  
&emsp;__inputs__: _pomdp_, a POMDP with states _S_, actions _A_(_s_), transition model _P_(_s&prime;_ &vert; _s_, _a_),  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;sensor model _P_(_e_ &vert; _s_), rewards _R_(_s_), discount _&gamma;_  
&emsp;&emsp;&emsp;&emsp;&emsp;_&epsi;_, the maximum error allowed in the utility of any state  
&emsp;__local variables__: _U_, _U&prime;_, sets of plans _p_ with associated utility vectors _&alpha;<sub>p</sub>_  

&emsp;_U&prime;_ &larr; a set containing just the empty plan \[\], with _&alpha;<sub>\[\]</sub>_(_s_) = _R_(_s_)  
&emsp;__repeat__  
&emsp;&emsp;&emsp;_U_ &larr; _U&prime;_  
&emsp;&emsp;&emsp;_U&prime;_ &larr; the set of all plans consisting of an action and, for each possible next percept,  
&emsp;&emsp;&emsp;&emsp;&emsp;a plan in _U_ with utility vectors computed according to Equation(__??__)  
&emsp;&emsp;&emsp;_U&prime;_ &larr; REMOVE\-DOMINATED\-PLANS(_U&prime;_)  
&emsp;__until__ MAX\-DIFFERENCE(_U_, _U&prime;_) &lt; _&epsi;_(1 &minus; _&gamma;_) &frasl; _&gamma;_  
&emsp;__return__ _U_  

---
__Figure ??__ A high\-level sketch of the value iteration algorithm for POMDPs. The REMOVE\-DOMINATED\-PLANS step and MAX\-DIFFERENCE test are typically implemented as linear programs.
