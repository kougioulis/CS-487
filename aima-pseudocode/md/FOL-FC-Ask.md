# FOL-FC-ASK

## AIMA3e
__function__ FOL-FC-ASK(_KB_, _&alpha;_) __returns__ a substitution or _false_  
&emsp;__inputs__: _KB_, the knowledge base, a set of first order definite clauses  
&emsp;&emsp;&emsp;&emsp;&emsp;_&alpha;_, the query, an atomic sentence  
&emsp;__local variables__: _new_, the new sentences inferred on each iteration

&emsp;__repeat until__ _new_ is empty  
&emsp;&emsp;&emsp;_new_ &larr; { }  
&emsp;&emsp;&emsp;__for each__ _rule_ __in__ _KB_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;(_p_<sub>1</sub> &and; ... &and; _p_<sub>n</sub> &rArr; _q_) &larr; STANDARDIZE-VARAIBLES(_rule_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ _&theta;_ such that SUBST(_&theta;_, _p_<sub>1</sub> &and; ... &and; _p_<sub>n</sub>) = SUBST(_&theta;_, _p_'<sub>1</sub> &and; ... &and; _p_'<sub>n</sub>)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;for some _p_'<sub>1</sub> &and; ... &and; _p_'<sub>n</sub> in _KB_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_q_' &larr; SUBST(_&theta;_, _q_)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _q_' does not unify with some sentence already in _KB_ or _new_ __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;add _q_' to _new_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_&Phi;_ _&larr;_ UNIFY(_q_', _&alpha;_)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _&Phi;_ is not _fail_ __then return__ _&Phi;_  
&emsp;&emsp;&emsp;add _new_ to _KB_  
&emsp;__return__ _false_  

---
__Figure__ ?? A conceptually straightforward, but very inefficient forward-chaining algorithm. On each iteration, it adds to _KB_ all the atomic sentences that can be inferred in one step from the implication sentences and the atomic sentences already in _KB_. The function STANDARDIZE-VARIABLES replaces all variables in its arguments with new ones that have not been used before.