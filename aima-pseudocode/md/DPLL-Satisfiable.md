# DPLL-SATISFIABLE?

## AIMA3e
__function__ DPLL-SATISFIABLE?(_s_) __returns__ _true_ or _false_  
&emsp;__inputs__: _s_, a sentence in propositional logic.  

&emsp;_clauses_ &larr; the set of clauses in the CNF representation of _s_  
&emsp;_symbols_ &larr; a list of the proposition symbols in _s_  
&emsp;__return__ DPLL(_clauses_, _symbols_, { })  

---
__function__ DPLL(_clauses_, _symbols_, _model_) __returns__ _true_ or _false_  

&emsp;__if__ every clause in _clauses_ is true in _model_ __then return__ _true_  
&emsp;__if__ some clause in _clauses_ is false in _model_ __then return__ _false_  
&emsp;_P_, _value_ &larr; FIND\-PURE\-SYMBOL(_symbols_, _clauses_, _model_)  
&emsp;__if__ _P_ is non-null __then return__ DPLL(_clauses_, _symbols_ - _P_, _model_ &cup; {_P_ = _value_})  
&emsp;_P_, _value_ &larr; FIND\-UNIT\-CLAUSE(_clauses_, _model_)  
&emsp;__if__ _P_ is non-null __then return__ DPLL(_clauses_, _symbols_ - _P_, _model_ &cup; {_P_ = _value_})  
&emsp;_P_ &larr; FIRST(_symbols_); _rest_ &larr; REST(_symbols_)  
&emsp;__return__ DPLL(_clauses_, _rest_, _model_ &cup; {_P_ = _true_}) __or__  
&emsp;&emsp;&emsp;&emsp;DPLL(_clauses_, _rest_, _model_ &cup; {_P_ = _false_})

---
__Figure__ ?? The DPLL algorithm for checking satisfiability of a sentence in propositional logic. The ideas behind FIND\-PURE\-SYMBOL and FIND\-UNIT\-CLAUSE are described in the text; each returns a symbol (or null) and the truth value to assign to that symbol. Like TT\-ENTAILS?, DPLL operates over partial models.
