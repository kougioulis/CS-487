# MINIMAL-CONSISTENT-DET

## AIMA3e
__function__ Minimal-Consistent-Det(_E_, _A_) __returns__ a set of attributes  
&emsp;__inputs__: _E_, a set of examples  
&emsp;&emsp;&emsp;&emsp;&emsp;_A_, a set of attributes, of size _n_  

&emsp;__for__ _i_ = 0 __to__ _n_ __do__  
&emsp;&emsp;&emsp;__for each__ subset _A<sub>i</sub>_ of _A_ of size _i_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ Consistent-Det?(_A<sub>i</sub>_, _E_) __then return__ _A<sub>i</sub>_  

---
__function__ Consistent-Det?(_A_, _E_) __returns__ a truth value  
&emsp;__inputs__: _A_, a set of attributes  
&emsp;&emsp;&emsp;&emsp;&emsp;_E_, a set of examples  
&emsp;__local variables__: _H_, a hash table  

&emsp;__for each__ example _e_ __in__ _E_ __do__  
&emsp;&emsp;&emsp;__if__ some example in _H_ has the same values as _e_ for the attributes _A_  
&emsp;&emsp;&emsp;&emsp;but a different classification __then return__ _false_  
&emsp;&emsp;&emsp;store the class of _e_ in_H_, indexed by the values for attributes _A_ of the example _e_  
&emsp;__return__ _true_  

---
__Figure ??__ An algorithm for finding a minimal consistent determination.
