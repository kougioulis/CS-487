# CROSS-VALIDATION-WRAPPER

## AIMA3e
__function__ CROSS-VALIDATION-WRAPPER(_Learner_, _k_, _examples_) __returns__ a hypothesis  
&emsp;__local variables__: _errT_, an array, indexed by _size_, storing training\-set error rates  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_errV_, an array, indexed by _size_, storing validation\-set error rates  
&emsp;__for__ _size_ = 1 __to__ &infin; __do__  
&emsp;&emsp;&emsp;_errT_\[_size_\], _errV_\[_size_\] &larr; CROSS\-VALIDATION(_Learner_, _size_, _k_, _examples_)  
&emsp;&emsp;&emsp;__if__ _errT_ has converged __then do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_best\_size_ &larr; the value of _size_ with minimum _errV_\[_size_\]  
&emsp;&emsp;&emsp;&emsp;&emsp;__return__ _Learner_(_best\_size_, _examples_)  

---
__function__ CROSS\-VALIDATION(_Learner_, _size_, _k_, _examples_) __returns__ two values:  
&emsp;_fold\_errT_ &larr; 0; _fold\_errV_ &larr; 0  
&emsp;__for__ _fold_ = 1 __to__ _k_ __do__  
&emsp;&emsp;&emsp;_training\_set_, _validation\_set_ &larr; PARTITION(_examples_, _fold_, _k_)  
&emsp;&emsp;&emsp;_h_ &larr; _Learner_(_size_, _training\_set_)  
&emsp;&emsp;&emsp;_fold\_errT_ &larr; _fold\_errT_ &plus; ERROR\-RATE(_h_, _training\_set_)  
&emsp;&emsp;&emsp;_fold\_errV_ &larr; _fold\_errV_ &plus; ERROR\-RATE(_h_, _validation\_set_)  
&emsp;__return__ _fold\_errT_ &frasl; _k_, _fold\_errV_ &frasl; _k_  

---  

Figure?? An algorithm to select the model that has the lowest error rate on validation data by building models of increasing complexity, and choosing the one with best empirical error rate on validation data. Here _errT_ means error rate on the training data, and _errV_ means error rate on the validation data. _Learner_(_size_, _exmaples_) returns a hypothesis whose complexity is set by the parameter _size_, and which is trained on the _examples_. PARTITION(_examples_, _fold_, _k_) splits _examples_ into two subsets: a validation set of size _N_ &frasl; _k_ and a training set with all the other examples. The split is different for each value of _fold_.  

---

In the fourth edition, cross vaidation wrapper has been renamed to Model-Selection.
