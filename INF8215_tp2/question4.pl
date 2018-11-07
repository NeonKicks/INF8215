%% Question4.pl

%%===================================
%% ----- Program for person(X) ------
%%===================================
ask(X) :-
	format('Is the person a ~w? ', [X]),
	read(Reply),
	Reply = 'yes'.

person(X) :- ask_leader(X).

ask_leader(X) :- ask('leader of some sorts'), leader(X).

leader(X) :- ask('political leader'), political_leader(X).
leader(X) :- ask('spiritual leader'), spiritual_leader(X).

%% People with precise characteristics as predicates
political_leader(mikhail_gorbachev).
political_leader(joseph_staline).
political_leader(dwight_d_eisenhower).
political_leader(richard_nixon).
political_leader(cleopatra).
prophet(jesus).
prophet(moses).
spiritual_leader(pope_francis).

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


%%===================================
%% ----- Program for object(X) ------
%%===================================
