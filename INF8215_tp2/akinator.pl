ask(governs, Y) :-
	format('~w governs? ', [Y]),
	read(Reply),
	Reply = 'yes'.

ask(musician, X) :-
	format('~w is a musician? ', [X]),
	read(Reply),
	Reply = 'yes'.

ask(X) :-
	format('Is the person a ~w? ', [X]),
	read(Reply),
	Reply = 'yes'.

person(X) :- politician(X).
person(X) :- artist(X).

artist(X) :- ask(singer), singer(X).
artist(X) :- ask(musician), musician(X).

politician(X) :- governs(X, Y), country(Y), ask(governs, Y).

singer(celine_dion).
musician(john_lewis).
governs(stephen_harper,canada).
governs(barack_obama,usa).
country(canada).
country(usa).