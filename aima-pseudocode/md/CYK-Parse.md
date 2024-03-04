# CYK-PARSE

## AIMA3e
__function__ CYK-Parse(_words_, _grammar_) __returns__ _P_, a table of probabilities  
&emsp;_N_ &larr; Length(_words_)  
&emsp;_M_ &larr; the number of nonterminal symbols in _grammar_  
&emsp;_P_ &larr; an array of size [_M_, _N_, _N_], initially all 0  
&emsp;/\* insert lexical rules for each word \*/  
&emsp;__for__ _i_ = 1 __to__ _N_ __do__  
&emsp;&emsp;__for__ __each__ rule of form (_X_ &rarr; _words<sub>i</sub>_[_P_]) __do__  
&emsp;&emsp;&emsp;_P_[_X_, _i_, 1] &larr; p  
&emsp;/\* Combine first and second parts of right-hand sides of rules, from short to long \*/  
&emsp;__for__ _length_ = 2 __to__ _N_ __do__  
&emsp;&emsp;__for__ _start_ = 1 __to__ _N_ - _length_ + 1 __do__  
&emsp;&emsp;&emsp;__for__ _len1_ = 1 __to__ _length_ - 1 __do__  
&emsp;&emsp;&emsp;&emsp;_len2_ &larr; _length_ - _len1_  
&emsp;&emsp;&emsp;&emsp;__for each__ rule of the form (_X_ &rarr; _Y_ _Z_ [_p_]) __do__  
&emsp;&emsp;&emsp;&emsp;&emsp;_P_[_X_, _start_, _length_] &larr; Max(_P_[_X_, _start_, _length_], _P_[_Y_, _start_, _len1_] x _P_[_Z_, _start_ + _len1_, _len2_] x _p_)  
&emsp;__return__ _P_  

---
__Figure ??__ The CYK algorithm for parsing. Given a sequence of words, it finds the most probable derivation for the whole sequence and for each subsequence. It returns the whole table, _P_, in which and entry _P_[_X_, _start_, _len_] is the probability of the most probable _X_ of length _len_ starting at position _start_. If there is no _X_ of that size at that location, the probability is 0.
