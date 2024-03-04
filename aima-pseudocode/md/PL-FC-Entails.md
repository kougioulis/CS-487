# PL-FC-ENTAILS?

## AIMA3e
__function__ PL-FC-ENTAILS?(_KB_, _q_) __returns__ _true_ or _false_  
&emsp;__inputs__: _KB_, the knowledge base, a set of propositional definite clauses  
&emsp;&emsp;&emsp;&emsp;&emsp;_q_, the query, a proposition symbol  
&emsp;_count_ &larr; a table, where _count_\[_c_\] is the number of symbols in _c_'s premise  
&emsp;_inferred_ &larr; a table, where _inferred_\[_s_\] is initially _false_ for all symbols  
&emsp;_agenda_ &larr; a queue of symbols, initially symbols known to be true in _KB_  

&emsp;__while__ _agenda_ is not empty __do__  
&emsp;&emsp;&emsp;_p_ &larr; POP(_agenda_)  
&emsp;&emsp;&emsp;__if__ _p_ = _q_ __then return__ _true_  
&emsp;&emsp;&emsp;__if__ _inferred_\[_p_\] = _false_ __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;_inferred_\[_p_\] &larr; _true_  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ clause _c_ in _KB_ where _p_ is in _c_.PREMISE __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;decrement _count_\[_c_\]  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _count_\[_c_\] = 0 __then__ add _c_.CONCLUSION to _agenda_  
&emsp;__return__ _false_  

--- 
__Figure__ ?? The forward-chaining algorithm for propositional logic. The _agenda_ keeps track of symbols known to be true but not yet "processed". The _count_ table keeps track of how many premises of each implication are as yet unknown. Whenever a new symbol _p_ from the agenda is processed, the count is reduced by one for each implication in whose premise _p_ appears (easily identified in constant time with appropriate indexing.) If a count reaches zero, all the premises of the implication are known, so its conclusion can be added to the agenda. Finally, we need to keep track of which symbols have been processed; a symbol that is already in the set of inferred symbols need not be added to the agenda again. This avoids redundant work and prevents loops caused by implications such as _P_ &rArr; _Q_ and _Q_ &rArr; _P_.
