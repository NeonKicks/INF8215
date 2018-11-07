%% Question4.pl

%%===================================
%% ----- Program for person(X) ------
%%===================================
ask(X) :-
	format('Is the person a ~w? ', [X]),
	read(Reply),
	Reply = 'yes'.

person(X) :- politician(X).
person(X) :- artist(X).

artist(X) :- ask(singer, X), singer(X).
artist(X) :- ask(musician, X), musician(X).

politician(X) :- governs(X, Y), country(Y), ask(governs, Y).

person(X) :- ask(person, X).
object(X) :- ask(object, X).

%% People with precise characteristics as predicates
political_leader(mikhail_gorbachev).
political_leader(joseph_staline).
political_leader(dwight_d_eisenhower).
political_leader(richard_nixon).
ruler(cleopatra).

director(quentin_tarantino).
actress(jennifer_lawrence).
actor(denzel_washington).
singer(michael_jackson).
singer(lady_gaga).

game_creator(hideo_kojima).
street_artist(banksy).
author(j_k_rowling).
author(victor_hugo).

driver(ayrton_senna).
driver(fernando_alonso).

game_character(lara_croft).
game_character(mario).
movie_character(james_bond).

prophet(jesus).
prophet(moses).
spiritual_leaderpope_francis).


%%===================================
%% ----- Program for object(X) ------
%%===================================