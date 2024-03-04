# REFLEX-VACUUM-AGENT

## AIMA3e
__function__ REFLEX-VACUUM-AGENT([_location,status_]) __returns__ an action  
&emsp;__if__ _status_ = _Dirty_ __then return__ _Suck_  
&emsp;__else if__ _location_ = A __then return__ _Right_  
&emsp;__else if__ _location_ = B __then return__ _Left_  

---
__Figure__ ?? The agent program for a simple reflex agent in the two-state vacuum environment. This program implements the agent function tabulated in Figure ??.
