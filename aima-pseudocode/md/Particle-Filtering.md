# PARTICLE-FILTERING

## AIMA3e
__function__ PARTICLE-FILTERING(__e__, _N_, _dbn_) __returns__ a set of samples for the next time step  
&emsp;__inputs__: __e__, the new incoming evidence  
&emsp;&emsp;&emsp;&emsp;&emsp;_N_, the number of samples to be maintained  
&emsp;&emsp;&emsp;&emsp;&emsp;_dbn_, a DBN with prior __P__(__X<sub>0</sub>__), transition model __P__(__X<sub>1</sub>__ &vert; __X<sub>0</sub>__), sensor model __P__(__E<sub>1</sub>__ &vert; __X<sub>1</sub>__)  
&emsp;__persistent__: _S_, a vector of samples of size _N_, initially generated from __P__(__X<sub>0</sub>__)  
&emsp;__local variables__: _W_, a vector of weights of size _N_  

&emsp;__for__ _i_ = 1 to _N_ __do__  
&emsp;&emsp;&emsp;_S_\[_i_\] &larr; sample from __P__(__X<sub>1</sub>__ &vert; __X<sub>0</sub>__ = _S_\[_i_\])&emsp;/\* step 1 \*/  
&emsp;&emsp;&emsp;_W_\[_i_\] &larr; __P__(__e__ &vert; __X<sub>1</sub>__ = _S_\[_i_\])&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;/\* step 2 \*/  
&emsp;_S_ &larr; WEIGHTED\-SAMPLE\-WITH\-REPLACEMENT(_N_, _S_, _W_)&emsp;&emsp;/\* step 3 \*/  
&emsp;__return__ _S_  

---
__Figure ??__ The particle filtering algorithm implemented as a recursive update operation with state (the set of samples). Each of the sampling operations involves sampling the relevant slice variables in topological order, much as in PRIOR\-SAMPLE. The  WEIGHTED\-SAMPLE\-WITH\-REPLACEMENT operation can be implemented to run in _O_(_N_) expected time. The step numbers refer to the description in the text.
