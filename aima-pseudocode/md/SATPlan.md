# SATPLAN

## AIMA3e
__function__ SATPLAN(_init_, _transition_, _goal_, _T_<sub>max</sub>) __returns__ solution or failure  
&emsp;__inputs__: _init_, _transition_, _goal_, constitute a description of the problem  
&emsp;&emsp;&emsp;&emsp;_T_<sub>max</sub>, an upper limit for plan length  

&emsp;__for__ _t_ = 0 to _T_<sub>max</sub> __do__  
&emsp;&emsp;&emsp;_cnf_ &larr; TRANSLATE-TO-SAT(_init_, _transition_, _goal_, _t_)  
&emsp;&emsp;&emsp;_model_ &larr; SAT-SOLVER(_cnf_)  
&emsp;&emsp;&emsp;__if__ _model_ is not null __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;__return__ EXTRACT-SOLUTION(_model_)  
&emsp;__return__ _failure_  

---
__Figure__ ?? The SATPlan algorithm. The planning problem is translated into a CNF sentence in which the goal is asserted to hold at a fixed time step _t_ and axioms are included for each time step up to _t_. If the satisfiability algorithm finds a model, then a plan is extracted by looking at those proposition symbols that refer to actions and are assigned _true_ in the model. If no model exists, then the process is repeated with the goal moved one step later. 