# TABLE-DRIVEN-AGENT

## AIMA3e
__function__ TABLE-DRIVEN-AGENT(_percept_) __returns__ an action  
&emsp;__persistent__: _percepts_, a sequence, initially empty  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_table_, a table of actions, indexed by percept   sequences, initially fully specified  

&emsp;append _percept_ to the end of _percepts_  
&emsp;_action_ &larr; LOOKUP(_percepts_, _table_)  
&emsp;__return__ _action_

---
__Figure__ ?? The TABLE-DRIVEN-AGENT  program is invoked for each new percept and returns an action each time. It retains the complete percept sequence in memory.
