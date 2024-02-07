# Random Forest

__function__ RANDOMFOREST( S,F ) __return__ solution:  
  &emsp;__input__ S --> training set consisting of (Xi,Yi)  
  &emsp;&emsp;F --> No. of features  
  &emsp;&emsp;H <-- null  
  
  &emsp;for __each__ i 1,..B(no.of trees) __do__  
    &emsp;&emsp;&emsp;h(i) --> RANDOMISEDTREE( S(i),F )  
    &emsp;&emsp;&emsp;H <-- H U {h(i)}  
    &emsp;&emsp;&emsp;__return__ H
    
__function__ RANDOMISEDTREE(S,F) __return__ learnt_tree :

  &emsp;for each node of the tree do  
    &emsp;&emsp;&emsp;f <-- Randomised subset of F  
    &emsp;&emsp;&emsp;Split and choose best feature in f
    
    
    
    
    
      
