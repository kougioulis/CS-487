# BLOCKS-WORLD

## AIMA3e
_Init_(_On_(_A_, _Table_) &and; _On_(_B_, _Table_) &and; _On_(_C_, _A_)  
&emsp;&emsp;&and; _Block_(_A_) &and; _Block_(_B_) &and; _Block_(_C_) &and; _Clear_(_B_) &and; _Clear_(_C_))  
_Goal_(_On_(_A_, _B_) &and; _On_(_B_, _C_))  
_Action_(_Move_(_b_, _x_, _y_),  
&emsp;PRECOND: _On_(_b_, _x_) &and; _Clear_(_b_) &and; _Clear_(_y_) &and; _Block_(_b_) &and; _Block_(_y_) &and;  
&emsp;&emsp;&emsp;&emsp;&emsp;(_b_ &ne; _x_) &and; (_b_ &ne; _y_) &and; (_x_ &ne; _y_),  
&emsp;EFFECT: _On_(_b_, _y_) &and; _Clear_(_x_) &and; &not; _On_(_b_, _x_) &and; &not; _Clear_(_y_))  
_Action_(_MoveToTable_(_b_, _x_),  
&emsp;PRECOND: _On_(_b_, _x_) &and; _Clear_(_b_) &and; _Block_(_b_) &and; (_b_ &ne; _x_),  
&emsp;EFFECT: _On_(_b_, _Table_) &and; _Clear_(_x_) &and; &not; _On_(_b_, _x_))

---
__Figure ??__ A planning problem in the blocks world: building a three-block tower. One solution is the sequence [_MoveToTable_(_C_, _A_), _Move_(_B_, _Table_, _C_), _Move_(_A_, _Table_, _B_)].
