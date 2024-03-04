# FOL-BC-ASK

## AIMA3e
__function__ FOL-BC-ASK(_KB_, _query_) __returns__ a generator of substitutions  
&emsp;__return__ FOL-BC-OR(_KB_, _query_, { })  

__generator__ FOL-BC-OR(_KB_, _goal_, _&theta;_) __yields__ a substitution  
&emsp;__for each__ rule (_lhs_ &rArr; _rhs_) __in__ FETCH-RULES-FOR-GOAL(_KB_, _goal_) __do__  
&emsp;&emsp;&emsp;(_lhs_, _rhs_) &larr; STANDARDIZE-VARIABLES((_lhs_, _rhs_))  
&emsp;&emsp;&emsp;__for each__ _&theta;_' __in__ FOL-BC-AND(_KB_, _lhs_, UNIFY(_rhs_, _goal_, _&theta;_)) __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;__yield__ _&theta;_'  

__generator__ FOL-BC-AND(_KB_, _goals_, _&theta;_) __yields__ a substitution  
&emsp;__if__ _&theta;_ = _failure_ __then return__  
&emsp;__else if__ LENGTH(_goals_) = 0 __then yield__ _&theta;_  
&emsp;__else do__  
&emsp;&emsp;&emsp;_first_, _rest_ &larr; FIRST(_goals_), REST(_goals_)  
&emsp;&emsp;&emsp;__for each__ _&theta;_' in FOL-BC-OR(_KB_, SUBST(_&theta;_, _first_), _&theta;_) __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ _&theta;_'' in FOL-BC-AND(_KB_, _rest_, _&theta;_') __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__yield__ _&theta;_''

---
__Figure__ ?? A simple backward-chaining algorithm for first-order knowledge bases.
