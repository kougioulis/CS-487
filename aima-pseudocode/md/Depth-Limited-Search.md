# DEPTH-LIMITED-SEARCH

## AIMA4e  

__function__ DEPTH-LIMITED-SEARCH(_problem_, _l_) __returns__ a solution, or failure, or cutoff  
&emsp;_frontier_ &larr; a FIFO queue initially containing one path, for the _problem_'s initial state  
&emsp;_solution_ &larr; failure  
&emsp;__while__  _frontier_ is not empty __do__  
&emsp;&emsp;&emsp;_parent_ &larr; pop(_frontier_)  
&emsp;&emsp;&emsp;__if__ depth(_parent_) > l __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;_solution_ &larr; cutoff  
&emsp;&emsp;&emsp;__else__  
&emsp;&emsp;&emsp;&emsp;&emsp;__for__ _child_ __in__ successors(_parent_) __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _child_ is a goal __then__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__return__ _child_  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;add _child_ to __frontier__  
&emsp;__return__  _solution_  

---
__Figure 3.14__ An implementation of depth-limited tree search. The algorithm has two dif-
ferent ways to signal failure to find a solution: it returns failure when it has exhausted all
paths and proved there is no solution at any depth, and returns cutoff to mean there might be
a solution at a deeper depth than l. Note that this algorithm does not keep track of reached
states, and thus might visit the same state multiple times on different paths.

## AIMA3e
__function__ DEPTH-LIMITED-SEARCH(_problem_,_limit_) __returns__ a solution, or failure/cutoff  
&emsp;__return__ RECURSIVE\-DLS(MAKE\-NODE(_problem_.INITIAL\-STATE),_problem_,_limit_)  

__function__ RECURSIVE\-DLS(_node_,_problem_,_limit_) __returns__ a solution, or failure/cutoff  
&emsp;__if__ _problem_.GOAL-TEST(_node_.STATE) __then return__ SOLUTION(_node_)  
&emsp;__else if__ _limit_ = 0 __then return__ _cutoff_  
&emsp;__else__  
&emsp;&emsp;&emsp;_cutoff\_occurred?_ &larr; false  
&emsp;&emsp;&emsp;__for each__ _action_ __in__ _problem_.ACTIONS(_node_.STATE) __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_child_ &larr; CHILD\-NODE(_problem_,_node_,_action_)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;_result_ &larr; RECURSIVE\-DLS(_child_,_problem_,_limit_\-1)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__if__ _result_ = _cutoff_ __then__ _cutoff\_occurred?_ &larr; true  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;__else if__ _result_ &ne; _failure_ __then return__ _result_  
&emsp;&emsp;&emsp;__if__ _cutoff\_occurred?_ __then return__ _cutoff_ __else return__ _failure_  

---
__Figure__ ?? A recursive implementation of depth\-limited tree search.
