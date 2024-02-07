# SPARE-TIRE-PROBLEM

## AIMA3e
_Init_(_Tire_(_Flat_) &and; _Tire_(_Spare_) &and; _At_(_Flat_, _Axle_) &and; _At_(_Spare_, _Trunk_))  
_Goal_(_At_(_Spare_, _Axle_))  
_Action_(_Remove_(_obj_, _loc_),  
&emsp;PRECOND: _At_(_obj_, _loc_)  
&emsp;EFFECT: &not; _At_(_obj_, _loc_) &and; _At_(_obj_, _Ground_))  
_Action_(_PutOn_(_t_, _Axle_),  
&emsp;PRECOND: _Tire_(_t_) &and; _At_(_t_, _Ground_) &and; &not; _At_(_Flat_, _Axle_)  
&emsp;EFFECT: &not; _At_(_t_, _Ground_) &and; _At_(_t_, _Axle_))  
_Action_(_LeaveOvernight_,  
&emsp;PRECOND:  
&emsp;EFFECT: &not; _At_(_Spare_, _Ground_) &and; &not; _At_(_Spare_, _Axle_) &and; &not; _At_(_Spare_, _Trunk_)  
&emsp;&emsp;&emsp;&emsp;&emsp;&and; &not; _At_(_Flat_, _Ground_) &and; &not; _At_(_Flat_, _Axle_) &and; &not; _At_(_Flat_, _Trunk_))

---
__Figure ??__ The simple spare tire problem.
