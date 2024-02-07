# VERSION-SPACE-LEARNING

## AIMA3e
__function__ Version-Space-Learning(_examples_) __returns__ a version space  
&emsp;__local variables__: _V_, the version space: the set of all hypotheses  

&emsp;_V_ &larr; the set of all hypotheses  
&emsp;__for each__ example _e_ in _examples_ __do__  
&emsp;&emsp;&emsp;__if__ _V_ is not empty __then__ _V_ &larr; Version-Space-Update(_V_, _e_)  
&emsp;__return__ _V_  

---
__function__ Version-Space-Update(_V_, _e_) __returns__ an updated version space  
&emsp;_V_ &larr; \{_h_ &isin; _V_ : _h_ is consistent with _e_\}  

---
__Figure ??__ The version space learning algorithm. It finds a subset of _V_ that is consistent with all the _examples_.
