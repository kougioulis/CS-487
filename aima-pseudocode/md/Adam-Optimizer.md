# ADAM-OPTIMIZER

## AIMA4e

__function__ ADAM-OPTIMIZER(_f_,_L_,***&theta;***,_&rho;_,_&alpha;_,_&delta;_) __returns__ updated ***&theta;***  
&emsp;&emsp;&emsp; /\* _Defualts:_ _&rho;<sub>1</sub>_ = 0.9; _&rho;<sub>2</sub>_ = 0.999;_&alpha;_ = 0.001;_&delta;_ = 10<sup>-8</sup> \*/  
&emsp;***s*** &larr; __0__  
&emsp;***r*** &larr; __0__  
&emsp; t &larr; 0  
&emsp;&emsp;&emsp;__while__ ***&theta;*** has not converged  
&emsp;&emsp;&emsp;&emsp;&emsp; ***x***, ***y*** &larr; a minibatch of _m_ examples from training set  
&emsp;&emsp;&emsp;&emsp;&emsp; ***g*** &larr; <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\frac{1}{m}\nabla_{\theta}\sum_{i}L(f(\textbf{x}^{(i)};\boldsymbol{\theta}),\textbf{y}^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{1}{m}\nabla_{\theta}\sum_{i}L(f(\textbf{x}^{(i)};\boldsymbol{\theta}),\textbf{y}^{(i)})" title="\frac{1}{m}\nabla_{\theta}\sum_{i}L(f(\textbf{x}^{(i)};\boldsymbol{\theta}),\textbf{y}^{(i)})" /></a> /\* compute gradient \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;_t_ &larr; _t_ &plus; 1   
&emsp;&emsp;&emsp;&emsp;&emsp;***s*** &larr; _&rho;<sub>1</sub>_***s*** &plus; (1 &minus; _&rho;<sub>1</sub>_)***g*** /\* _Update biased first moment estimate_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;***r*** &larr; _&rho;<sub>2</sub>_***r*** &plus; (1 &minus; _&rho;<sub>2</sub>_)***g***<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\odot" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\odot" title="\odot" /></a>***g*** /\*_Update biased second moment estimate_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{\textbf{s}}\gets\frac{\textbf{s}}{1-\rho_1^t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{\textbf{s}}\gets\frac{\textbf{s}}{1-\rho_1^t}" title="\hat{\textbf{s}}\gets\frac{\textbf{s}}{1-\rho_1^t}" /></a>  /\* _Correct bias in first moment_ \*/   
&emsp;&emsp;&emsp;&emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{\textbf{r}}\gets\frac{\textbf{r}}{1-\rho_2^t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{\textbf{r}}\gets\frac{\textbf{r}}{1-\rho_2^t}" title="\hat{\textbf{r}}\gets\frac{\textbf{r}}{1-\rho_2^t}" /></a> /\* _Correct bias in second moment_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\boldsymbol{\Delta}\boldsymbol{\theta}&space;=&space;-\epsilon\frac{\hat{\textbf{s}}}{\sqrt{\hat{\textbf{r}}}&plus;\delta}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\Delta}\boldsymbol{\theta}&space;=&space;-\epsilon\frac{\hat{\textbf{s}}}{\sqrt{\hat{\textbf{r}}}&plus;\delta}" title="\boldsymbol{\Delta}\boldsymbol{\theta} = -\epsilon\frac{\hat{\textbf{s}}}{\sqrt{\hat{\textbf{r}}}+\delta}" /></a>  /\* _Compute update (operations applied element-wise)_ \*/  
&emsp;&emsp;&emsp;&emsp;&emsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\boldsymbol{\theta}&space;\gets&space;\boldsymbol{\theta}&space;&plus;&space;\boldsymbol{\Delta}\boldsymbol{\theta}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\theta}&space;\gets&space;\boldsymbol{\theta}&space;&plus;&space;\boldsymbol{\Delta}\boldsymbol{\theta}" title="\boldsymbol{\theta} \gets \boldsymbol{\theta} + \boldsymbol{\Delta}\boldsymbol{\theta}" /></a> /\* _Apply update_ \*/   


