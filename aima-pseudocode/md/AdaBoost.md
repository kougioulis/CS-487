# ADABOOST

## AIMA3e
__function__ ADABOOST(_examples_, _L_, _K_) __returns__ a weighted\-majority hypothesis  
&emsp;__inputs__: _examples_, set of _N_ labeled examples (_x<sub>1</sub>_, _y<sub>1</sub>_),&hellip;,(_x<sub>N</sub>_,_y<sub>N</sub>_)  
&emsp;&emsp;&emsp;&emsp;_L_, a learning algorithm  
&emsp;&emsp;&emsp;&emsp;_K_, the number of hypotheses in the ensemble  
&emsp;__local variables__: __w__, a vector of _N_ example weights, initially 1 &frasl; _N_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__h__, a vector of _K_ hypotheses  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__z__, a vector of _K_ hypothesis weights  

&emsp;__for__ _k_ = 1 __to__ _K_ __do__  
&emsp;&emsp;&emsp;__h__\[_k_\] &larr; _L_(_examples_, __w__)  
&emsp;&emsp;&emsp;_error_ &larr; 0  
&emsp;&emsp;&emsp;__for__ _j_ = 1 __to__ _N_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ __h__\[_k_\](_x<sub>j</sub>_) &ne; _y<sub>j</sub>_ __then__ _error_ &larr; _error_ &plus; __w__\[_j_\]  
&emsp;&emsp;&emsp;__for__ _j_ = 1 __to__ _N_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ __h__\[_k_\](_x<sub>j</sub>_) = _y<sub>j</sub>_ __then__ __w__\[_j_\] &larr; __w__\[_j_\] &middot; _error_ &frasl; (1 &minus; _error_)  
&emsp;&emsp;&emsp;__w__ &larr; NORMALIZE(__w__)  
&emsp;&emsp;&emsp;__Z__\[_k_\] &larr; log(1 &minus; _error_) &frasl; _error_  
&emsp; __return__ WEIGHTED\-MAJORITY(__h__, __z__)  

---
__Figure ??__ The ADABOOST variant of the boosting method for ensemble learning. The algorithm generates hypothesis by successively reweighting the training examples. The function WEIGHTED\-MAJORITY generates a hypothesis that returns the output value with the highest vote from the hypotheses in __h__, with votes weighted by __z__.
