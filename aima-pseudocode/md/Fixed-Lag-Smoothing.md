# FIXED-LAG-SMOOTHING

## AIMA3e
__function__ FIXED-LAG-SMOOTHING(_e<sub>t</sub>_, _hmm_, _d_) __returns__ a distribution over __X__<sub>_t_&minus;_d_</sub>  
&emsp;__inputs__: _e<sub>t</sub>_, the current evidence for time step _t_  
&emsp;&emsp;&emsp;&emsp;&emsp;_hmm_, a hidden Markov model with _S_ &times; _S_ transition matrix __T__  
&emsp;&emsp;&emsp;&emsp;&emsp;_d_, the length of the lag for smoothing  
&emsp;__persistent__: _t_, the current time, initially 1  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__f__, the forward message __P__(_X<sub>t</sub>_ &vert; _e_<sub>1:_t_</sub>), initially _hmm_.PRIOR  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__B__, the _d_\-step backward transformation matrix, initially the identity matrix  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_e<sub>t&minus;d:t<sub>_, double\-ended list of evidence from _t_ &minus; _d_ to _t_, initially empty  
&emsp;__local variables__: __O__<sub>_t_&minus;_d_</sub>, __O__<sub>_t_</sub>, diagonal matrices containing the sensor model information  

&emsp;add _e<sub>t</sub>_ to the end of _e<sub>t&minus;d:t<sub>_  
&emsp;__O__<sub>_t_</sub> &larr; diagonal matrix containing __P__(_e<sub>t</sub>_ &vert; _X<sub>t</sub>_)  
&emsp;__if__ _t_ &gt; _d_ __then__  
&emsp;&emsp;&emsp;__f__ &larr; FORWARD(__f__, _e<sub>t</sub>_)  
&emsp;&emsp;&emsp;  remove  _e_<sub>_t_&minus;_d_&minus;1</sub> from the beginning of _e<sub>t&minus;d:t<sub>_  
&emsp;&emsp;&emsp;__O__<sub>_t_&minus;_d_</sub> &larr; diagonal matrix containing __P__(_e<sub>t&minus;d</sub>_ &vert; _X<sub>t&minus;d</sub>_)  
&emsp;&emsp;&emsp;__B__ &larr; __O__<sub>_t_&minus;_d_</sub><sup>&minus;1</sup>__T__<sup>&minus;1</sup>  __BTO__<sub>_t_</sub>  
&emsp;__else__ __B__ &larr; __BTO__<sub>_t_</sub>  
&emsp;_t_ &larr; _t_ &plus; 1  
&emsp;__if__ _t_ &gt; _d_ __then return__ NORMALIZE(__f__ &times; __B1__) __else return__ null  

---
__Figure ??__ An algorithm for smoothing with a fixed time lag of _d_ steps, implemented as an oline algorithm that outputs the new smoothed estimate given the observation for a new time step. Notice that the final output NORMALIZE(__f__ &times; __B1__) is just &alpha; __f__ &times; __b__, by Equation (__??__).
