# REJECTION-SAMPLING

## AIMA3e
__function__ REJECTION-SAMPLING(_X_, __e__, _bn_, _N_) __returns__ an estimate of __P__(_X_ &vert; __e__)  
&emsp;__inputs__: _X_, the query variable  
&emsp;&emsp;&emsp;&emsp;&emsp;__e__, observed values for variables __E__  
&emsp;&emsp;&emsp;&emsp;&emsp;_bn_, a Bayesian network  
&emsp;&emsp;&emsp;&emsp;&emsp;_N_, the total number of samples to be generated  
&emsp;__local variables__: __N__, a vector of counts for each value of _X_, initially zero  

&emsp;__for__ _j_ = 1 to _N_ __do__  
&emsp;&emsp;&emsp;__x__ &larr; PRIOR\-SAMPLE(_bn_)  
&emsp;&emsp;&emsp;__if x__ is consistent with __e then__  
&emsp;&emsp;&emsp;&emsp;&emsp;__N__\[_x_\] &larr; __N__\[_x_\] &plus; 1 where _x_ is the value of _X_ in __x__  
&emsp;__return__ NORMALIZE(__N__)  

---
__Figure__ ?? The rejection\-sampling algorithm for answering queries given evidence in a Bayesian network.
