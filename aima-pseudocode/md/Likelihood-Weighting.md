# LIKELIHOOD-WEIGHTING

## AIMA3e
__function__ LIKELIHOOD-WEIGHTING(_X_, __e__, _bn_, _N_) __returns__ an estimate of __P__(_X_ &vert; __e__)  
&emsp;__inputs__: _X_, the query variable  
&emsp;&emsp;&emsp;&emsp;&emsp;__e__, observed values for variables __E__  
&emsp;&emsp;&emsp;&emsp;&emsp;_bn_, a Bayesian network specifying joint distribution __P__(_X<sub>1</sub>_, &hellip;, _X<sub>n</sub>_)  
&emsp;&emsp;&emsp;&emsp;&emsp;_N_, the total number of samples to be generated  
&emsp;__local variables__: __W__, a vector of weighted counts for each value of _X_, initially zero  

&emsp;__for__ _j_ = 1 to _N_ __do__  
&emsp;&emsp;&emsp;__x__, _w_ &larr; WEIGHTED\-SAMPLE(_bn_, __e__)  
&emsp;&emsp;&emsp;__W__\[_x_\] &larr; __W__\[_x_\] &plus; _w_ where _x_ is the value of _X_ in __x__  
&emsp;__return__ NORMALIZE(__W__)  

---
__function__ WEIGHTED\-SAMPLE(_bn_, __e__) __returns__ an event and a weight  
&emsp;_w_ &larr; 1; __x__ &larr; an event with _n_ elements initialized from __e__  
&emsp;__foreach__ variable _X<sub>i</sub>_ __in__ _X<sub>1</sub>_, &hellip;, _X<sub>n</sub>_ __do__  
&emsp;&emsp;&emsp;__if__ _X<sub>i</sub>_ is an evidence variable with value _x<sub>i</sub>_ in __e__  
&emsp;&emsp;&emsp;&emsp;&emsp;__then__ _w_ &larr; _w_ &times; _P_(_X<sub>i</sub>_ = _x<sub>i</sub>_ &vert; _parents_(_X<sub>i</sub>_))  
&emsp;&emsp;&emsp;&emsp;&emsp;__else__ __x__\[i\] &larr; a random sample from __P__(_X<sub>i</sub>_ &vert; _parents_(_X<sub>i</sub>_))  
&emsp;__return__ __x__, _w_  

---
__Figure__ ?? The likelihood\-weighting  algorithm for inference in Bayesian networks. In WEIGHTED\-SAMPLE, each nonevidence variable is sampled according to the conditional distribution given the values already sampled for the variable's parents, while a weight is accumulated based on the likelihood for each evidence variable.
