# UNIFY

## AIMA3e
__function__ UNIFY(_x_, _y_, _&theta;_) __returns__ a substitution to make _x_ and _y_ identical  
&emsp;__inputs__: _x_, a variable, constant, list, or compound  
&emsp;&emsp;&emsp;&emsp;_y_, a variable, constant, list, or compound  
&emsp;&emsp;&emsp;&emsp;_&theta;_, the substitution built up so far (optional, defaults to empty)  

&emsp;__if__ _&theta;_ = failure __then return__ failure  
&emsp;__else if__ _x_ = _y_ __then return__ _&theta;_  
&emsp;__else if__ VARIABLE?(_x_) __then return__ UNIVY-VAR(_x_, _y_, _&theta;_)  
&emsp;__else if__ VARIABLE?(_y_) __then return__ UNIFY-VAR(_y_, _x_, _&theta;_)  
&emsp;__else if__ COMPOUND?(_x_) and COMPOUND?(_y_) __then__  
&emsp;&emsp;&emsp;__return__ UNIFY(_x_.ARGS, _y_.ARGS, UNIFY(_x_.OP, _y_.OP, _&theta;_))  
&emsp;__else if__ LIST?(_x_) __and__ LIST?(_y_) __then__  
&emsp;&emsp;&emsp;__return__ UNIFY(_x_.REST, _y_.REST, UNIFY(_x_.FIRST, _y_.FIRST, _&theta;_))  
&emsp;__else return__ failure

---
__function__ UNIFY-VAR(_var_, _x_, _&theta;_) __returns__ a substitution

&emsp;__if__ {_var_ / _val_} &isin; _&theta;_ __then return__ UNIFY(_val_, _x_, _&theta;_)  
&emsp;__else if__ {_x_ / _val_} &isin; _&theta;_ __then return__ UNIFY(_var_, _val_, _&theta;_)  
&emsp;__else if__ OCCUR-CHECK?(_var_, _x_) __then return__ failure  
&emsp;__else return__ add {_var_/_x_} to _&theta;_

---
__Figure__ ?? The unification algorithm. The algorithm works by comparing the structures of the inputs, elements by element. The substitution _&theta;_ that is the argument to UNIFY is built up along the way and is used to make sure that later comparisons are consistent with bindings that were established earlier. In a compound expression, such as F(A, B), the OP field picks out the function symbol F and the ARGS field picks out the argument list (A, B).