# MODEL-SELECTION

## AIMA4e  
__function__ MODEL-SELECTION(_Learner_, _examples_, _k_) __returns__ a hypothesis  
&emsp;__local variables__: _err_, an array, indexed by _size_, storing validation\-set error rates  
&emsp;__for__ _size_ = 1 __to__ &infin; __do__   
&emsp;&emsp;&emsp;_err_\[_size_\] &larr; CROSS\-VALIDATION(_Learner_, _size_, _examples_, _k_)  
&emsp;&emsp;&emsp;__if__ _err_ is starting to increase significantly __then do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_best\_size_ &larr; the value of _size_ with minimum _err_\[_size_\]  
&emsp;&emsp;&emsp;&emsp;&emsp;__return__ _Learner_(_best\_size_, _examples_)  

---
__function__ CROSS\-VALIDATION(_Learner_, _size_, _examples_, _k_) __returns__ average training set error rate:  
&emsp;_errs_ &larr; 0  
&emsp;__for__ _fold_ = 1 __to__ _k_ __do__  
&emsp;&emsp;&emsp;_training\_set_, _validation\_set_ &larr; PARTITION(_examples_, _fold_, _k_)  
&emsp;&emsp;&emsp;_h_ &larr; _Learner_(_size_, _training\_set_)  
&emsp;&emsp;&emsp;_errs_ &larr; _errs_ &plus; ERROR\-RATE(_h_, _validation\_set_)    
&emsp;__return__ _errs_ &frasl; _k_   // average error rate on validation sets, across k-fold cross-validation 

---  

Figure?? An algorithm to select the model that has the lowest error rate on validation data by building models of increasing complexity, and choosing the one with best empirical error rate, _err_, on validation data. _Learner_(_size_, _exmaples_) returns a hypothesis whose complexity is set by the parameter _size_, and which is trained on the _examples_. PARTITION(_examples_, _fold_, _k_) splits _examples_ into two subsets: a validation set of size _N_ &frasl; _k_ and a training set with all the other examples. The split is different for each value of _fold_.  

---

In the fourth edition, cross vaidation wrapper has been renamed to Model-Selection.

