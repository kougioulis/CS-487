# SIMPLEX-REFLEX-AGENT

## AIMA3e
__function__ SIMPLE-REFLEX-AGENT(_percept_) __returns__ an action  
&emsp;__persistent__: _rules_, a set of condition\-action rules  

&emsp;_state_ &larr; INTERPRET-INPUT(_percept_)  
&emsp;_rule_ &larr; RULE-MATCH(_state_,_rules_)  
&emsp;_action_ &larr; _rule_.ACTION  
&emsp;__return__ _action_  

---
__Figure__ ?? A simple reflex agent. It acts according to a rule whose condition matches the current state, as defined by the percept.
