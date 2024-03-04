# ELIMINATION-ASK

## AIMA3e
__function__ ELIMINATION-ASK(_X_, __e__, _bn_) __returns__ a distribution over _X_  
&emsp;__inputs__: _X_, the query variable  
&emsp;&emsp;&emsp;&emsp;&emsp;__e__, observed values for variables __E__  
&emsp;&emsp;&emsp;&emsp;&emsp;_bn_, a Bayesian network specifying joint distribution __P__(_X<sub>1</sub>_, &hellip;, _X<sub>n</sub>_)  

&emsp;_factors_ &larr; \[\]  
&emsp;__for each__ _var_ __in__ ORDER(_bn_.VARS) __do__  
&emsp;&emsp;&emsp;_factors_ &larr; \[MAKE\-FACTOR(_var_, __e__) &vert; _factors_\]  
&emsp;&emsp;&emsp;__if__ _var_ is a hidden variable __then__ _factors_ &larr; SUM\-OUT(_var_, _factors_)  
&emsp;__return__ NORMALIZE(POINTWISE\-PRODUCT(_factors_))  

---
__Figure__ ?? The variable elimination algorithm for inference in Bayesian networks.
