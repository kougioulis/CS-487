# WALKSAT

## AIMA3e
__function__ WALKSAT(_clauses_, _p_, _max\_flips_) __returns__ a satisfying model or _failure_  
&emsp;__inputs__: _clauses_, a set of clauses in propositional logic  
&emsp;&emsp;&emsp;&emsp;_p_, the probability of choosing to do a "random walk" move, typically around 0.5  
&emsp;&emsp;&emsp;&emsp;_max\_flips_, number of flips allowed before giving up  

&emsp;_model_ &larr; a random assignment of _true_/_false_ to the symbols in _clauses_  
&emsp;__for__ _i_ = 1 to _max\_flips_ __do__  
&emsp;&emsp;&emsp;__if__ _model_ satisfies _clauses_ __then return__ _model_  
&emsp;&emsp;&emsp;_clause_ &larr; a randomly selected clause from _clauses_ that is false in _model_  
&emsp;&emsp;&emsp;__with probability__ _p_ flip the value in _model_ of a randomly selected symbol from _clause_  
&emsp;&emsp;&emsp;__else__ flip whichever symbol in _clause_ maximizes the number of satisfied clauses  
&emsp;__return__ _failure_  

---
__Figure__ ?? The WALKSAT algorithm for checking satisfiability by randomly flipping the values of variables. Many versions of the algorithm exist.