# MODEL-BASED-REFLEX-AGENT

## AIMA3e
__function__ MODEL-BASED-REFLEX-AGENT(_percept_) __returns__ an action  
&emsp;__persistent__: _state_, the agent's current conception of the world state  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_model_, a description of how the next state depends on current state and action  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_rules_, a set of condition\-action rules  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_action_, the most recent action, initially none

&emsp;_state_ &larr; UPDATE-STATE(_state_, _action_, _percept_, _model_)  
&emsp;_rule_ &larr; RULE-MATCH(_state_,_rules_)  
&emsp;_action_ &larr; _rule_.ACTION  
&emsp;__return__ _action_  

---
__Figure__ ?? A model-based reflex agent. It keeps track of the current state of the world, using an internal model. It then chooses an action in the same way as the reflex agent.
