# ENUMERATION-ASK

## AIMA3e
__function__ ENUMERATION-ASK(_X_, __e__, _bn_) __returns__ a distribution over _X_  
&emsp;__inputs__: _X_, the query variable  
&emsp;&emsp;&emsp;&emsp;&emsp;__e__, observed values for variables __E__  
&emsp;&emsp;&emsp;&emsp;&emsp;_bn_, a Bayes net with variables \{_X_\} &Union; __E__ &Union; __Y__ /\* __Y__ = hidden variables \*/  

&emsp;__Q__(_X_) &larr; a distribution over _X_, initially empty  
&emsp;__for each__ value _x<sub>i</sub>_ of _X_ __do__  
&emsp;&emsp;&emsp;__Q__(_x<sub>i</sub>_) &larr; ENUMERATE\-ALL(_bn_.VARS, __e__<sub>_x_<sub>_i_</sub></sub>)  
&emsp;&emsp;&emsp;&emsp;&emsp;where __e__<sub>_x_<sub>_i_</sub></sub> is __e__ extended with _X_ = _x<sub>i</sub>_  
&emsp;__return__ NORMALIZE(__Q__(_X_))  

---
__function__ ENUMERATE\-ALL(_vars_, __e__) __returns__ a real number  
&emsp;__if__ EMPTY?(_vars_) __then return__ 1.0  
&emsp;_Y_ &larr; FIRST(_vars_)  
&emsp;__if__ _Y_ has value _y_ in __e__  
&emsp;&emsp;&emsp;__then return__ _P_(_y_ &vert; _parents_(_Y_)) &times; ENUMERATE\-ALL(REST(_vars_), __e__)  
&emsp;&emsp;&emsp;__else return__ &sum;<sub>_y_</sub> _P_(_y_ &vert; _parents_(_Y_)) &times; ENUMERATE\-ALL(REST(_vars_), __e__<sub>_y_</sub>)  
&emsp;&emsp;&emsp;&emsp;&emsp;where __e__<sub>_y_</sub> is __e__ extended with _Y_ = _y_  

---
__Figure__ ?? The enumeration algorithm for answering queries on Bayesian networks.
