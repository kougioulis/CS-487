# SIMPLE-PROBLEM-SOLVING-AGENT

## AIMA3e
__function__ SIMPLE-PROBLEM-SOLVING-AGENT(_percept_) __returns__ an action  
&emsp;__persistent__: _seq_, an action sequence, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_state_, some description of the current world state  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_goal_, a goal, initially null  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_problem_, a problem formulation

&emsp;_state_ &larr; UPDATE-STATE(_state_, _percept_)  
&emsp;__if__ _seq_ is empty __then__  
&emsp;&emsp;&emsp;_goal_ &larr; FORMULATE-GOAL(_state_)  
&emsp;&emsp;&emsp;_problem_ &larr; FORMULATE-PROBLEM(_state_, _goal_)  
&emsp;&emsp;&emsp;_seq_ &larr; SEARCH(_problem_)  
&emsp;&emsp;&emsp;__if__ _seq_ = _failure_ __then return__ a null action  
&emsp;_action_ &larr; FIRST(_seq_)  
&emsp;_seq_ &larr; REST(_seq_)  
&emsp;__return__ _action_  

---
__Figure__ ?? A simple problem-solving agent. It first formulates a goal and a problem, searches for a sequence of actions that would solve the problem, and then executes the actions one at a time. When this is complete, it formulates another goal and starts over.
