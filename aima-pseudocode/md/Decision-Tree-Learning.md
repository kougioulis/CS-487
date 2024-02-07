# DECISION-TREE-LEARNING

## AIMA3e
__function__ DECISION-TREE-LEARNING(_examples_, _attributes_, _parent\_examples_) __returns__ a tree  
&emsp;__if__ _examples_ is empty __then return__ PLURALITY\-VALUE(_parent\_examples_)  
&emsp;__else if__ all _examples_ have the same classification __then return__ the classification  
&emsp;__else if__ _attributes_ is empty __then return__ PLURALITY\-VALUE(_examples_)  
&emsp;__else__  
&emsp;&emsp;&emsp;_A_ &larr; argmax<sub>_a_ &isin; _attributes_</sub> IMPORTANCE(_a_, _examples_)  
&emsp;&emsp;&emsp;_tree_ &larr; a new decision tree with root test _A_  
&emsp;&emsp;&emsp;__for each__ value _v<sub>k</sub>_ of _A_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_exs_ &larr; \{ _e_ : _e_ &isin; _examples_ __and__ _e_._A_ = _v<sub>k</sub>_ \}  
&emsp;&emsp;&emsp;&emsp;&emsp;_subtree_ &larr; DECISION-TREE-LEARNING(_exs_, _attributes_ &minus; _A_, _examples_)  
&emsp;&emsp;&emsp;&emsp;&emsp;add a branch to _tree_ with label \(_A_ = _v<sub>k</sub>_\) and subtree _subtree_  
&emsp;&emsp;&emsp;__return__ _tree_  

---
__Figure ??__ The decision\-tree learning algorithm. The function  IMPORTANCE is described in Section __??__. The function PLURALITY\-VALUE selects the most common output value among a set of examples, breaking ties randomly.
