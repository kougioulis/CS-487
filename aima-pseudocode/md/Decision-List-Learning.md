# DECISION-LIST-LEARNING

## AIMA3e
__function__ DECISION-LIST-LEARNING(_examples_) __returns__ a decision list, or _failure_  
&emsp;__if__ _examples_ is empty __then return__ the trivial decision list _No_  
&emsp;_t_ &larr; a test that matches a nonempty subset _examples<sub>t</sub>_ of _examples_  
&emsp;&emsp;&emsp;such that the members of _examples<sub>t</sub>_ are all positive or all negative  
&emsp;__if__ there is no such _t_ __then return__ _failure_  
&emsp;__if__ the examples in _examples<sub>t</sub>_ are positive __then__ _o_ &larr; _Yes_ __else__ _o_ &larr; _No_  
&emsp;__return__ a decision list with initial test _t_ and outcome _o_ and remaining tests given by  
&emsp;&emsp;&emsp;&emsp;DECISION-LIST-LEARNING(_examples_ &minus; _examples<sub>t</sub>_)  

---
__Figure ??__ An algorithm for learning decision lists.
