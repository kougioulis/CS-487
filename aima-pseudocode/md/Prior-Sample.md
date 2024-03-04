# PRIOR-SAMPLE

## AIMA3e
__function__ PRIOR-SAMPLE(_bn_) __returns__ an event sampled from the prior specified by _bn_  
&emsp;__inputs__: _bn_, a Bayesian network specifying joint distribution __P__(_X<sub>1</sub>_, &hellip;, _X<sub>n</sub>_)  

&emsp;__x__ &larr; an event with _n_ elements  
&emsp;__foreach__ variable _X<sub>i</sub>_ __in__ _X<sub>1</sub>_, &hellip;, _X<sub>n</sub>_ __do__  
&emsp;&emsp;&emsp;_x_\[_i_\] &larr; a random sample from __P__(_X<sub>i</sub>_ &vert; _parents_(_X<sub>i</sub>_))  
&emsp;__return x__

---
__Figure__ ?? A sampling algorithm that generates events from a Bayesian network. Each variable is sampled according to the conditional distribution given the values already sampled for the variable's parents.
