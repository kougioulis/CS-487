# BACK-PROP-LEARNING-WITH-REGULARIZATION

## AIMA3e
__function__ BACK-PROP-LEARNING-WITH-REGULARIZATION(_examples_, _network_) __returns__ a neural network  
&emsp;__inputs__ _examples_, a set of examples, each with input vector __x__ and output vector __y__  
&emsp;&emsp;&emsp;&emsp;_network_, a multilayer network with _L_ layers, weights _w<sub>i,j</sub>_, activation function _g_, regularization parameter &lambda; 
&emsp;__local variables__: &Delta;, a vector of errors, indexed by network node  

/* &Phi; is a function of the weights. It depends on the type of regularization, e.g., for _L<sub>2</sub>_ regularization, &Phi;=&Sigma;<sub>_j_</sub> ||_w<sub>i,j</sub>_||<sup>_2_</sup> */  

&emsp;__repeat__  
&emsp;&emsp;&emsp;__for each__ weight _w<sub>i,j</sub>_ in _network_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_w<sub>i,j</sub>_ &larr; a small random number  
&emsp;&emsp;&emsp;__for each__ example (__x__, __y__) __in__ _examples_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;/\* _Propagate the inputs forward to compute the outputs_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ node _i_ in the input layer __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_a<sub>i</sub>_ &larr; _x<sub>i</sub>_  
&emsp;&emsp;&emsp;&emsp;&emsp;__for__ _l_ = 2 __to__ _L_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ node _j_ in layer _l_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_in<sub>j</sub>_ &larr; &Sigma;<sub>_i_</sub> _w<sub>i,j</sub>_ _a<sub>i</sub>_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_a<sub>j</sub>_ &larr; _g_(_in<sub>j</sub>_)  
&emsp;&emsp;&emsp;&emsp;&emsp;/\* _Propagate deltas backward from output layer to input layer_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ node _j_ in the output layer __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&Delta;\[_j_\] &larr; _g_&prime;(_in<sub>j</sub>_) &times; (_y<sub>i</sub>_ &minus; _a<sub>j</sub>_)  
&emsp;&emsp;&emsp;&emsp;&emsp;__for__ _l_ = _L_ &minus; 1 __to__ 1 __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ node _i_ in layer _l_ __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&Delta;\[_i_\] &larr; _g_&prime;(_in<sub>i</sub>_) &Sigma;<sub>_j_</sub> _w<sub>i,j</sub>_ &Delta;\[_j_\]  + &lambda; &Phi;(_w<sub>i,j</sub>_)
&emsp;&emsp;&emsp;&emsp;&emsp;/\* _Update every weight in network using deltas_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;__for each__ weight _w<sub>i,j</sub>_ in _network_ __do__ 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_w<sub>i,j</sub>_ &larr; _w<sub>i,j</sub>_ &plus; _&alpha;_ &times; _a<sub>i</sub>_ &times; &Delta;\[_j_\]  
 &emsp;__until__ some stopping criterion is satisfied  
 &emsp;__return__ _network_  

---
__Figure ??__ The back\-propagation algorithm for learning in multilayer networks.
