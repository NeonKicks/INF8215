%%_Question4.pl
ask(musician, X) :-
	format('~w is a musician? ', [X]),
	read(Reply),
	Reply = 'yes'.
      
ask(object, Y) :-
      format('~w governs? ', [Y]),
      read(Reply),
      Reply = 'yes'.

person(X) :- ask(person, X).
object(X) :- ask(object, X).


people([
michael_jackson,
mikhail_gorbachev,
jennifer_lawrence,
hideo_kojima,
banksy,
lara_croft,
mario,
j_k_rowling,
lady_gaga,
quentin_tarantino,
joseph_staline,
dwight_d_eisenhower,
cleopatra,
victor_hugo,
jesus,
ayrton_senna,
moses,
fernando_alonso,
pope_francis,
james_bond,
denzel_washington,
richard_nixon]).