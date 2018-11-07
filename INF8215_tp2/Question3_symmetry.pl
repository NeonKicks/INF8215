%% Exercise 3

%% x is a prerequisite course for y is noted prerequisite(x,y)
prerequisite(null,'INF1005C').
prerequisite(null,'INF1500').
prerequisite(null,'LOG2810').
prerequisite(null,'MTH1007').
prerequisite(null,'LOG2990').
prerequisite('INF1005C','INF1010').
prerequisite('INF1005C','LOG1000').
prerequisite('INF1005C','INF1600').
prerequisite('INF1500','INF1600').
prerequisite('LOG1000','LOG2410').
prerequisite('INF1010','LOG2410').
prerequisite('INF1010','INF2010').
prerequisite('INF2010','INF2705').
%% x is a corequisite course for y is noted corequisite(x,y)
corequisite('LOG2810','INF2010').
corequisite('MTH1007','INF2705').
corequisite('LOG2990','INF2705').
corequisite('INF1600','INF1900').
corequisite('INF2205','INF1900').
corequisite('LOG1000','INF1900').

corequisites(A,B) :- corequisite(A,B).
corequisites(A,B) :- corequisite(B,A).

%% requisite(A,B) returns the different requisite courses A needed for B, with duplicates
requisite(A, B) :- prerequisite(A, B) ; corequisites(A,B).
requisite(A, B) :- prerequisite(A, X), requisite(X, B) ; corequisites(A, X), requisite(X, B).

%% completeRequirementsFor(C,List2) returns List2 of required courses for C with no duplicates
completeRequirementsFor(C,List2) :- setof(X,requisite(X,C),List1), delete(List1,null,List2).
