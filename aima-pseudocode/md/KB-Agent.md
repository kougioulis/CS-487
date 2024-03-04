# KB-AGENT

## AIMA3e
__function__ KB-AGENT(_percept_) __returns__ an _action_  
&emsp;__persistent__: _KB_, a knowledge base  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_t_, a counter, initially 0, indicating time  

&emsp;TELL(_KB_, MAKE\-PERCEPT\-SENTENCE(_percept_, _t_))  
&emsp;_action_ &larr; ASK(_KB_, MAKE\-ACTION\-QUERY(_t_))  
&emsp;TELL(_KB_, MAKE\-ACTION\-SENTENCE(_action_, _t_))  
&emsp;_t_ &larr; _t_ + 1  
&emsp;__return__ _action_  

---
__Figure__ ?? A generic knowledge\-based agent. Given a percept, the agent adds the percept to its knowledge base, asks the knowledge base for the best action, and tells the knowledge base that it has in fact taken that action.
