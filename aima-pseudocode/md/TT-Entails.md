# TT-ENTAILS

## AIMA3e
__function__ TT-ENTAILS?(_KB_, _&alpha;_) __returns__ _true_ or _false_  
&emsp;__inputs__: _KB_, the knowledge base, a sentence in propositional logic  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_&alpha;_, the query, a sentence in propositional logic  

&emsp;_symbols_ &larr; a list of the propositional symbols in _KB_ and _&alpha;_  
&emsp;__return__ TT-CHECK-ALL(_KB_, _&alpha;_, _symbols_, { })  

---
__function__ TT-CHECK-ALL(_KB_, _&alpha;_, _symbols_, _model_) __returns__ _true_ or _false_  
&emsp;__if__ EMPTY?(_symbols_) __then__  
&emsp;&emsp;&emsp;__if__ PL-TRUE?(_KB_, _model_) __then return__ PL-TRUE?(_&alpha;_, _model_)  
&emsp;&emsp;&emsp;__else return__ _true_&emsp;_//_&emsp;_when KB is false, always return true_  
&emsp;__else do__  
&emsp;&emsp;&emsp;_P_ &larr; FIRST(_symbols_)  
&emsp;&emsp;&emsp;_rest_ &larr; REST(_symbols_)  
&emsp;&emsp;&emsp;__return__(TT-CHECK-ALL(_KB_, _&alpha;_, _rest_, _model_ &cup; { _P_ = _true_ })  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__and__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;TT-CHECK-ALL(_KB_, _&alpha;_, _rest_, _model_ &cup; { _P_ = _false_ }))  
   

---
__Figure__ ?? A truth-table enumeration algorithm for deciding propositional entailment. (TT stands for truth table.) PL-TRUE? returns _true_ if a sentence holds within a model. The variable _model_ represents a partial model \- an assignment to some of the symbols. The keyword "__and__" is used here as a logical operation on its two arguments, returning _true_ or _false_.
