# GIBBS-ASK

## AIMA3e
__function__ GIBBS-ASK(_X_, __e__, _bn_, _N_) __returns__ an estimate of __P__(_X_ &vert; __e__)  
&emsp;__local variables__: __N__, a vector of counts for each value of _X_, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__Z__, the nonevidence variables in _bn_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__x__, the current state of the network, initially copied from __e__  

&emsp;initialize __x__ with random values for the variables in __Z__  
&emsp;__for__ _j_ = 1 to _N_ __do__  
&emsp;&emsp;&emsp;__for each__ _Z<sub>i</sub>_ in __Z__ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;set the value of _Z<sub>i</sub>_ in __x__ by sampling from __P__(_Z<sub>i</sub>_ &vert; _mb_(_Z<sub>i</sub>_))  
&emsp;&emsp;&emsp;&emsp;&emsp;__N__\[_x_\] &larr; __N__\[_x_\] &plus; 1 where _x_ is the value of _X_ in __x__  
&emsp;__return__ NORMALIZE(__N__)  

---
__Figure__ ?? The Gibbs sampling algorithm for approximate inference in Bayesian networks; this version cycles through the variables, but choosing variables at random also works.  

---
  
## AIMA4e  
__function__ GIBBS-ASK(_X_, __e__, _bn_, _N_) __returns__ an estimate of __P__(_X_ &vert; __e__)  
&emsp;__local variables__: __N__, a vector of counts for each value of _X_, initially zero  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__Z__, the nonevidence variables in _bn_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__x__, the current state of the network, initially copied from __e__  
  
&emsp;initialize __x__ with random values for the variables in __Z__  
&emsp;__for__ _j_ = 1 to _N_ __do__  
&emsp;&emsp;&emsp;__choose__ any variable _Z<sub>i</sub>_ from __Z__ acoording to any distribution _&rho;(i)_  
&emsp;&emsp;&emsp;&emsp;&emsp;set the value of _Z<sub>i</sub>_ in __x__ by sampling from __P__(_Z<sub>i</sub>_ &vert; _mb_(_Z<sub>i</sub>_))  
&emsp;&emsp;&emsp;&emsp;&emsp;__N__\[_x_\] &larr; __N__\[_x_\] &plus; 1 where _x_ is the value of _X_ in __x__  
&emsp;__return__ NORMALIZE(__N__)  

---
__Figure__ ?? The Gibbs sampling algorithm for approximate inference in Bayesian networks; this version cycles through the variables, but choosing variables at random also works.  
