# JOB-SHOP-SCHEDULING-PROBLEM

## AIMA3e
_Jobs({AddEngine1 ≺ AddWheels1 ≺ Inspect1 },_  
&emsp;_{AddEngine2 ≺ AddWheels2 ≺ Inspect2 })_  
_Resources(EngineHoists(1), WheelStations(1), Inspectors(2), LugNuts(500))_  
_Action(AddEngine1_, DURATION: 30,  
&emsp;USE: _EngineHoists(1))_  
_Action(AddEngine2_, DURATION: 60,  
&emsp;USE: _EngineHoists(1))_  
_Action(AddWheels1_, DURATION: 30,  
&emsp;CONSUME: _LugNuts(20),_ USE: _WheelStations(1))_  
_Action(AddWheels2_, DURATION: 15,  
&emsp;CONSUME: _LugNuts(20)_, USE: _WheelStations(1))_  
_Action(Inspect<sub>i</sub>_, DURATION: 10,  
&emsp;USE: _Inspectors(1))_

---
__Figure ??__ A job-shop scheduling problem for assembling two cars, with resource constraints. The
notation _A_ ≺ _B_ means that action _A_ must precede action _B_.
