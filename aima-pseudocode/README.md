# Pseudocode
Pseudocode descriptions of the algorithms from [Russell](http://www.cs.berkeley.edu/~russell/) and [Norvig's](http://www.norvig.com/) [Artificial Intelligence - A Modern Approach](http://aima.cs.berkeley.edu/).

The algorithms as they appear in the book are available in pdf format:
- [**algorithms.pdf**](https://github.com/aimacode/pseudocode/blob/master/aima4e-algorithms.pdf)

The files listed below give the same algorithms, but in markdown format. **We need help checking that the 4th edition versions are correct.** We are in the process of updating from the old  [3rd edition](https://github.com/aimacode/pseudocode/blob/master/aima3e-algorithms.pdf) algorithms. (_Note:_ when editing pseudocode in a `md/*.md` file, put **two spaces at the end of a line**; this keeps the line separate from the following line.)


| 3rd ed. | 4th ed. | Pseudo-code Algorithm|
|:------:|:------:|:---------------------|
| 2.3 | 2.7 | [TABLE-DRIVEN-AGENT](md/Table-Driven-Agent.md) |
| 2.4 | 2.8 | [REFLEX-VACUUM-AGENT](md/Reflex-Vacuum-Agent.md) |
| 2.6 | 2.10 | [SIMPLE-REFLEX-AGENT](md/Simple-Reflex-Agent.md) |
| 2.8 | 2.12 | [MODEL-BASED-REFLEX-AGENT](md/Model-Based-Reflex-Agent.md) |
| 3.1 | x | [SIMPLE-PROBLEM-SOLVING-AGENT](md/Simple-Problem-Solving-Agent.md) |
| 3.7 | 3.7 | [BEST-FIRST-SEARCH](md/Tree-Search-and-Graph-Search.md) |
| 3.11 |3.9  | [BREADTH-FIRST-SEARCH](md/Breadth-First-Search.md) |
| 3.17 | 3.12 | [ITERATIVE-DEEPENING-SEARCH](md/Iterative-Deepening-Search.md) |
| x    | 3.14 | [BIBF-SEARCH]() (Bidirectional Best-First) |
| 3.13 |  x| [UNIFORM-COST-SEARCH](md/Uniform-Cost-Search.md) |
| 3.16 |  x| [DEPTH-LIMITED-SEARCH](md/Depth-Limited-Search.md) |
| 3.24 | 3.22 | [RECURSIVE-BEST-FIRST-SEARCH](md/Recursive-Best-First-Search.md) |
| 4.2 | 4.2 | [HILL-CLIMBING](md/Hill-Climbing.md) |
| 4.5 | 4.4 | [SIMULATED-ANNEALING](md/Simulated-Annealing.md) |
| 4.8 | 4.7 | [GENETIC-ALGORITHM](md/Genetic-Algorithm.md) |
| 4.11 |4.10  | [AND-OR-GRAPH-SEARCH](md/And-Or-Graph-Search.md) |
| 4.21 |4.20  | [ONLINE-DFS-AGENT](md/Online-DFS-Agent.md) |
| 4.24 | 4.23 | [LRTA*-AGENT](md/LRTAStar-Agent.md) |
| 5.3 | 5.3 | [MINIMAX-SEARCH](md/Minimax-Decision.md) (was MINIMAX-DECISION in 3e) |
| 5.7 | 5.7 | [ALPHA-BETA-SEARCH](md/Alpha-Beta-Search.md) |
| x | 5.11 | [MONTE-CARLO-TREE-SEARCH](md/Monte-Carlo-Tree-Search.md) 
| 6.3 | 6.3 | [AC-3](md/AC-3.md) |
| 6.5 | 6.5 | [BACKTRACKING-SEARCH](md/Backtracking-Search.md) |
| 6.8 | 6.9 | [MIN-CONFLICTS](md/Min-Conflicts.md) |
| 6.11 | 6.11 | [TREE-CSP-SOLVER](md/Tree-CSP-Solver.md) |
| 7.1|  7.1 | [KB-AGENT](md/KB-Agent.md) |
| 7.8 | 7.10 | [TT-ENTAILS](md/TT-Entails.md) |
| 7.9 | 7.13 | [PL-RESOLUTION](md/PL-Resolution.md) |
| 7.12 |7.15  | [PL-FC-ENTAILS?](md/PL-FC-Entails.md) |
| 7.14 | 7.17 | [DPLL-SATISFIABLE?](md/DPLL-Satisfiable.md) |
| 7.15 | 7.18 | [WALKSAT](md/WalkSAT.md) |
| 7.17 | 7.20 | [HYBRID-WUMPUS-AGENT](md/Hybrid-Wumpus-Agent.md) |
| 7.19 | 7.22 | [SATPLAN](md/SATPlan.md) |
| 9.1 | 9.1 | [UNIFY](md/Unify.md) |
| 9.3 | 9.3 | [FOL-FC-ASK](md/FOL-FC-Ask.md) |
| 9.6 | 9.6 | [FOL-BC-ASK](md/FOL-BC-Ask.md) |
| 9.8 | 9.8 | [APPEND](md/Append.md) |
| 10.1 |11.1 | [AIR-CARGO-TRANSPORT-PROBLEM](md/Air-Cargo-Transport-Problem.md) |
| 10.2 |11.2 | [SPARE-TIRE-PROBLEM](md/Spare-Tire-Problem.md) |
| 10.3 |11.4 | [BLOCKS-WORLD](md/Blocks-World.md) |
| 10.7 |x | [HAVE-CAKE-AND-EAT-CAKE-TOO-PROBLEM](md/Have-Cake-And-Eat-Cake-Too.md) |
| 10.9 | x | [GRAPHPLAN](md/GraphPlan.md) |
| 11.4 |11.7 | [REFINEMENT-HIGH-LEVEL-ACTIONS](md/Refinement-High-Level-Actions.md)
| 11.5 | 11.8 | [HIERARCHICAL-SEARCH](md/Hierarchical-Search.md) |
| 11.8 |11.11  | [ANGELIC-SEARCH](md/Angelic-Search.md) |
| 11.1 |11.13 | [JOB-SHOP-SCHEDULING-PROBLEM](md/Job-Shop-Scheduling-Problem.md)
| 13.1 | 12.1 | [DT-AGENT](md/DT-Agent.md) |
| 14.9 |13.11  | [ENUMERATION-ASK](md/Enumeration-Ask.md) |
| 14.10 | 13.13 | [ELIMINATION-ASK](md/Elimination-Ask.md) |
| 14.12 |13.16  | [PRIOR-SAMPLE](md/Prior-Sample.md) |
| 14.13|13.17 | [REJECTION-SAMPLING](md/Rejection-Sampling.md) |
| 14.14 |13.18  | [LIKELIHOOD-WEIGHTING](md/Likelihood-Weighting.md) |
| 14.15 | 13.20 | [GIBBS-ASK](md/Gibbs-Ask.md) |
| 15.4 | 14.4 | [FORWARD-BACKWARD](md/Forward-Backward.md) |
| 15.6 | 14.6 | [FIXED-LAG-SMOOTHING](md/Fixed-Lag-Smoothing.md) |
| 15.17 | 14.17 | [PARTICLE-FILTERING](md/Particle-Filtering.md) |
| x |15.5 | [OUPM](md/oupm.md) (for citation extraction) |
| x | 15.6 | [NET-VISA](md/net-visa.md) |
| x | 15.9 | [RADAR](md/radar.md) (OUPM for radar tracking)|
| x | 15.11| [GENERATE-IMAGE](md/generate-image.md)|
| x |15.15| [GENERATE-MARKOV-LETTERS](md/generate-markov-letters.md)|
| 16.9 | 16.9 | [INFORMATION-GATHERING-AGENT](md/Information-Gathering-Agent.md) |
| 17.4 | 17.6 | [VALUE-ITERATION](md/Value-Iteration.md) |
| 17.7 | 17.9 | [POLICY-ITERATION](md/Policy-Iteration.md) |
| 17.9 |17.16  | [POMDP-VALUE-ITERATION](md/POMDP-Value-Iteration.md) |
| 11.10 | 18.1| [DOUBLES-TENNIS-PROBLEM](md/Doubles-Tennis-Problem.md) |
| 18.4 | 19.5 | [LEARN-DECISION-TREE](md/Decision-Tree-Learning.md) |
| 18.7 | 19.8 | [CROSS-VALIDATION-WRAPPER](md/Cross-Validation-Wrapper.md) |
| 18.10 | 19.11 | [DECISION-LIST-LEARNING](md/Decision-List-Learning.md) |
| 18.33 | 19.25 | [ADABOOST](md/AdaBoost.md) |
| 19.2 |  x| [CURRENT-BEST-LEARNING](md/Current-Best-Learning.md) |
| 19.3 |  x| [VERSION-SPACE-LEARNING](md/Version-Space-Learning.md) |
| 19.8 |  x| [MINIMAL-CONSISTENT-DET](md/Minimal-Consistent-Det.md) |
| 19.12 | x | [FOIL](md/Foil.md) |
| 21.2 | 22.2 | [PASSIVE-ADP-AGENT](md/Passive-ADP-Agent.md) |
| 21.4 | 22.4 | [PASSIVE-TD-AGENT](md/Passive-TD-Agent.md) |
| 21.8 | 22.8 | [Q-LEARNING-AGENT](md/Q-Learning-Agent.md) |
| 22.1 | x | [HITS](md/Hits.md) |
| 23.4 |23.5  | [CYK-PARSE](md/CYK-Parse.md) |
| 23.5 | 23.8| [SENTENCE-TREE](md/Sentence-Tree.md) |
| 25.9 | 26.6 | [MONTE-CARLO-LOCALIZATION](md/Monte-Carlo-Localization.md) |
| 29.1 | x | [POWERS-OF-2](md/Powers-Of-2.md) |
