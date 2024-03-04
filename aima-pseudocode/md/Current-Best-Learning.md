# CURRENT-BEST-LEARNING

## AIMA3e
__function__ Current-Best-Learning(_examples_, _h_) __returns__ a hypothesis or fail  
&emsp;__if__ _examples_ is empty __then__  
&emsp;&emsp;&emsp;__return__ _h_  
&emsp;_e_ &larr; First(_examples_)  
&emsp;__if__ _e_ is consistent with _h_ __then__  
&emsp;&emsp;&emsp;__return__ Current-Best-Learning(Rest(_examples_), _h_)  
&emsp;__else if__ _e_ is a false positive for _h_ __then__  
&emsp;&emsp;&emsp;__for each__ _h'_ __in__ specializations of _h_ consistent with _examples_ seen so far __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_h''_ &larr; Current-Best-Learning(Rest(_examples_), _h'_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _h''_ &ne; _fail_ __then return__ _h''_  
&emsp;__else if__ _e_ is a false negative for _h_ __then__  
&emsp;&emsp;&emsp;__for each__ _h'_ __in__ generalizations of _h_ consistent with _examples_ seen so far __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_h''_ &larr; Current-Best-Learning(Rest(_examples_), _h'_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _h''_ &ne; _fail_ __then return__ _h''_  
&emsp;__return__ _fail_  

---
__Figure ??__ The current-best-hypothesis learning algorithm. It searches for a consistent hypothesis that fits all the examples and backtracks when no consistent specialization/generalization can be found. To start the algorithm, any hypothesis can be passed in; it will be specialized or generalized as needed.
