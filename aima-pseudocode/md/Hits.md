# HITS

## AIMA3e
__function__ HITS(_query_) __returns__ _pages_ with hub and authority numbers  
&emsp;_pages_ &larr; Expand-Pages(Relevant-Pages(_query_))  
&emsp;__for each__ _p_ __in__ _pages_ __do__  
&emsp;&emsp;_p_.Authority &larr; 1  
&emsp;&emsp;_p_.Hub &larr; 1  
&emsp;__repeat until__ convergence __do__  
&emsp;&emsp;__for each__ _p_ __in__ _pages_ __do__  
&emsp;&emsp;&emsp;_p_.Authority &larr; &sum;<sub>i</sub> Inlink<sub>i</sub>(_p_).Hub  
&emsp;&emsp;&emsp;_p_.Hub &larr; &sum;<sub>i</sub> Outlink<sub>i</sub>(_p_).Authority  
&emsp;&emsp;Normalize(_pages_)  
&emsp;__return__ _pages_  

---
__Figure ??__ The HITS algorithm for computing hubs and authorities with respect to a query. Relevant-Pages fetches the pages that match the query, and Expand-Pages adds in every page that links to or is linked from one of the relevant pages. Normalize divides each page's score by the square root of sum of the squares of all pages' scores (separately for both the authority and hubs scores).
