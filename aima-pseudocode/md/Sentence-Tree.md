# SENTENCE-TREE

## AIMA3e

[ [S [NP-SBJ-2 Her eyes]  
&emsp;&emsp;[VP were  
&emsp;&emsp;&emsp;&emsp;[VP glazed  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[NP *-2]  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[SBAR-ADV as if  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[S [NP-SBJ she]  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[VP did n’t  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[VP [VP hear [NP *-1]]  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;or  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[VP [ADVP even] see [NP *-1]]  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[NP-1 him]]]]]]]]  
.]

---
__Figure ??__ Annotated tree for the sentence “Her eyes were glazed as if she didn’t hear or even
see him.” from the Penn Treebank. Note that in this grammar there is a distinction between an object
noun phrase (_NP_) and a subject noun phrase (_NP-SBJ_). Note also a grammatical phenomenon we have
not covered yet: the movement of a phrase from one part of the tree to another. This tree analyzes
the phrase “hear or even see him” as consisting of two constituent _VP_ s, [VP hear [NP *-1]] and [VP
[ADVP even] see [NP *-1]], both of which have a missing object, denoted *-1, which refers to the _NP_
labeled elsewhere in the tree as [NP-1 him].
