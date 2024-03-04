# AIR-CARGO-TRANSPORT-PROBLEM

## AIMA3e
_Init_(_At_(_C<sub>1</sub>_, _SFO_) &and; _At_(_C<sub>2</sub>_, _JFK_) &and; _At_(_P<sub>1</sub>_, _SFO_) &and; _At_(_P<sub>2<sub>_, _JFK_)  
&emsp;&emsp;&and; _Cargo_(_C<sub>1</sub>_) &and; _Cargo_(_C<sub>2</sub>_) &and; _Plane_(_P<sub>1</sub>_) &and; _Plane_(_P<sub>2</sub>_)  
&emsp;&emsp;&and; _Airport_(_JFK_) &and; _Airport_(_SFO_))  
_Goal_(_At_(_C<sub>1</sub>_, _JFK_) &and; _At_(_C<sub>2</sub>_, _SFO_))  
_Action_(_Load_(_c_, _p_, _a_),  
&emsp;PRECOND: _At_(_c_, _a_) &and; _At_(_p_, _a_) &and; _Cargo_(_c_) &and; _Plane_(_p_) &and; _Airport_(_a_)  
&emsp;EFFECT: &not; _At_(_c_, _a_) &and; _In_(_c_, _p_))  
_Action_(_Unload_(_c_, _p_, _a_),  
&emsp;PRECOND: _In_(_c_, _p_) &and; _At_(_p_, _a_) &and; _Cargo_(_c_) &and; _Plane_(_p_) &and; _Airport_(_a_)  
&emsp;EFFECT: _At_(_c_, _a_) &and; &not; _In_(_c_, _p_))  
_Action_(_Fly_(_p_, _from_, _to_),  
&emsp;PRECOND: _At_(_p_, _from_) &and; _Plane_(_p_) &and; _Airport_(_from_) &and; _Airport_(_to_)  
&emsp;EFFECT: &not; _At_(_p_, _from_) &and; _At_(_p_, _to_))

---
__Figure ??__ A PDDL description of an air cargo transportation planning problem.
