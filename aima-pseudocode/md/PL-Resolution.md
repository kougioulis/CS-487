# PL-RESOLUTION

## AIMA3e
__function__ PL-RESOLUTION(_KB_, _&alpha;_) __returns__ _true_ or _false_  
&emsp;__inputs__: _KB_, the knowledge base, a sentence in propositional logic  
&emsp;&emsp;&emsp;&emsp;&emsp;_&alpha;_, the query, a sentence in propositional logic  

&emsp;_clauses_ &larr; the set of clauses in the CNF representation of _KB_ &and; &not;_&alpha;_  
&emsp;_new_ &larr; { }  
&emsp;__loop do__  
&emsp;&emsp;&emsp;__for each__ pair of clauses _C<sub>i</sub>_, _C<sub>j</sub>_ __in__ _clauses_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_resolvents_ &larr; PL-RESOLVE(_C<sub>i</sub>_, _C<sub>j</sub>_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _resolvents_ contains the empty clause __then return__ _true_  
&emsp;&emsp;&emsp;&emsp;&emsp;_new_ &larr; _new_ &cup; _resolvents_  
&emsp;&emsp;&emsp;__if__ _new_ &sube; _clauses_ __then return__ _false_  
&emsp;&emsp;&emsp;_clauses_ &larr; _clauses_ &cup; _new_

---
Figure ?? A simple resolution algorithm for propositional logic. The function PL-RESOLVE returns the set of all possible clauses obtained by resolving its two inputs.