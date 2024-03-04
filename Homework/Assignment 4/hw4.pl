%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%   Homework 4  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CS-487 Fall Semester 2022-2023  %%%%
%%%% Department of Computer Science  %%%%
%%%%       University of Crete       %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Nikolaos Kougioulis (1285) %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
connect(a, b).

connect(b, c).

connect(b, d).

connect(e, d).

connect(d, c).

connect(c, f).

connect(f, e).

%interconnected(X, Y):- connect(X, Y).
%interconnected(X, Y):- connect(Y, X).
interconnected(X, Y):- connect(X, Y); connect(Y, X).

interconnected(a,b).
interconnected(b,d).
interconnected(e,d).
interconnected(d,c).
interconnected(c,f).
interconnected(f,e).

:- style_check(-singleton).

%exists_path/2
%exists_path(StartNode, EndNode).
%exists_path(StartNode, EndNode) :- connect(StartNode, NextNode), exists_path(NextNode, EndNode).

%exists_path/2 with interconnect
exists_path(StartNode, EndNode).
exists_path(StartNode, EndNode) :- interconnected(StartNode, NextNode), exists_path(NextNode, EndNode).

%path/3
path(StartNode, EndNode, Route) :- path(StartNode, EndNode, [], Route).

path(StartNode, StartNode, _, [StartNode]).

path(StartNode, EndNode, VisitedNodes, [StartNode|Nodes]) :-
    \+ member(StartNode, VisitedNodes), %member/2 is true if StartNode is member of VisitedNodes list
    dif(StartNode, EndNode), %dif/2 predicate is true if-f StartNode and EndNode are different
    interconnected(StartNode, Node),
    path(Node, EndNode, [StartNode|VisitedNodes], Nodes).

connect(a, b, 3).

connect(b, d, 4).

connect(b, c, 12).

connect(d, c, 7).

connect(d, e, 5).

connect(c, f, 11).

connect(e, f, 15).

interconnected(X, Y, C):- connect(X, Y, C).
interconnected(X, Y, C):- connect(Y, X, C).

interconnected(a, b, 3).
interconnected(b, d, 4).
interconnected(b, c, 12).
interconnected(d, c, 7).
interconnected(d, e, 5).
interconnected(c, f, 11).
interconnected(e, f, 15).

%cost_path/4
cost_path(StartNode, EndNode, Route, Cost) :-
    cost_path(StartNode, EndNode, [], Route, Cost).

cost_path(StartNode, StartNode, _, [StartNode], 0).

cost_path(StartNode, EndNode, VisitedNodes, [StartNode|Nodes], Cost) :-
    interconnected(StartNode, Node, C),
    \+ member(StartNode, VisitedNodes), %member/2 is true if StartNode is member of VisitedNodes list
    dif(StartNode, EndNode), %dif/2 predicate is true if-f StartNode and EndNode are different
    cost_path(Node, EndNode, [StartNode|VisitedNodes], Nodes, RCost),
    Cost is C+RCost.
