# INFORMATION-GATHERING-AGENT

## AIMA3e
__function__ INFORMATION-GATHERING-AGENT(_percept_) __returns__ an action  
&emsp;__persistent__: _D_, a decision network  

&emsp;integrate _percept_ into _D_  
&emsp;_j_ &larr; the value that maximizes _VPI_(_E<sub>j</sub>_) / _Cost_(_E<sub>j</sub>_)  
&emsp;__if__  _VPI_(_E<sub>j</sub>_) >  _Cost_(_E<sub>j</sub>_)  
&emsp;&emsp;&emsp;__return__ REQUEST(_E<sub>j</sub>_)  
&emsp;__else return__ the best action from _D_  

---
__Figure ??__ Design of a simple information-gathering agent. The agent works by repeatedly selecting the observation with the highest information value, until the cost of the next observation is greater than its expected benefit.
