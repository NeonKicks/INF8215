%% Question4.pl

%%===================================
%% ----- Program for person(X) ------
%%===================================
ask(X) :-
	format('Is the person ~w? ', [X]),
	read(Reply),
	Reply = 'yes'.


person(X) :- ask('a leader'), leader(X).
person(X) :- ask('fictional'), fictional(X).
person(X) :- ask('related to Hollywood'), hollywood(X).
person(X) :- ask('a singer'), sings(X).
person(X) :- ask('a creator of some sorts'), creator(X).
person(X) :- ask('a sportsperson'), sports(X).

leader(X) :- ask('a political leader'), political(X).
leader(X) :- ask('a spiritual leader'), spiritual(X).

political(X) :- ask('in power in 1953'), political_1953(X).
political(X) :- political_not_1953(X).

political_1953(X) :- political_leader_1953(X, Y), nationality(Y), ask(Y).
political_not_1953(X) :- political_leader(X, Y), nationality(Y), ask(Y).

spiritual(X) :- ask('a prophet'), spiritual_prophet(X); spiritual_leader(X).

spiritual_prophet(X) :- ask('the son of Mary'), prophet(X,yes); prophet(X,no).


fictional(X) :- ask('originally from a movie'), movie_character(X).
fictional(X) :- ask('originally from a video game'), game_char(X).

game_char(X) :- game_character(X, Y), sex(Y), ask(Y).

hollywood(X) :- ask('a movie director'), director(X).
hollywood(X) :- ask('an actress'), actress(X).
hollywood(X) :- ask('an actor'), actor(X).

sings(X) :- singer(X, Y), sex(Y), ask(Y).

creator(X) :- ask('a video game creator'), game_creator(X).
creator(X) :- ask('a street artist'), street_artist(X).
creator(X) :- ask('an author'), writer(X).
writer(X) :- author(X, Y), sex(Y), ask(Y).

sports(X) :- ask('a driver'), racer(X).
racer(X) :- driver(X), ask(X).

%%------------------- People with precise characteristics as predicates -------------------
%% Making rule for direct verification
political_leader(X) :- political_leader(X,Y), nationality(Y); political_leader_1953(X,Y), nationality(Y).
political_leader_1953(joseph_staline, russian).
political_leader_1953(dwight_d_eisenhower, american).
political_leader(mikhail_gorbachev, russian).
political_leader(richard_nixon, american).
political_leader(cleopatra, egyptian).
prophet(X) :- prophet(X,B), boolean(B).
prophet(jesus,yes).
prophet(moses,no).
spiritual_leader(pope_francis).

game_character(X) :- game_character(X,Y), sex(Y).
game_character(lara_croft,female).
game_character(mario,male).
movie_character(james_bond).

director(quentin_tarantino).
actress(jennifer_lawrence).
actor(denzel_washington).
singer(X) :- singer(X,Y), sex(Y).
singer(michael_jackson,male).
singer(lady_gaga,female).

game_creator(hideo_kojima).
street_artist(banksy).
author(X) :- author(X,Y), sex(Y).
author(j_k_rowling,female).
author(victor_hugo,male).

driver(ayrton_senna).
driver(fernando_alonso).

nationality(russian).
nationality(american).
nationality(egyptian).

boolean(yes).
boolean(no).

sex(male).
sex(female).

%%===================================
%% ----- Program for object(X) ------
%%===================================
askobj(X) :-
	format('Is the object ~w? ', [X]),
	read(Reply),
	Reply = 'yes'.

askfunction(X) :-
	format('Does the object ~w? ', [X]),
	read(Reply),
	Reply = 'yes'.

object(X) :- askfunction('use electricity'), electric(X).
object(X) :- askobj('used to clean things'), cleaning(X).
object(X) :- askobj('something nearly all students have on them'), students(X).
object(X) :- askobj('usually found in kitchen cabinets'), kitchen_obj(X).
object(X) :- askobj('a type of furniture'), furnitures(X).
object(X) :- askobj('a musical instrument'), instrument(X).
object(X) :- askobj('a plant'), plant(X).
object(X) :- askobj('a type of thin sheet'), sheet(X).


electric(X) :- askobj('used for making food or beverages'), food(X).
electric(X) :- askobj('used to clean things'), e_cleaning_device(X).
electric(X) :- askfunction('have a screen'), screen_device(X).
electric(X) :- askobj('used to light things'), lighting(X).

food(X) :- askobj('used to cook meals'), cooking(X); not_cooking(X).

cooking(X) :- askobj('used to saute or fry things'), appliance(X,yes); appliance(X,no).
not_cooking(X) :- askfunction('prepare beverages'), small_appliance(X,yes); small_appliance(X,no).

screen_device(X) :- askfunction('fit in a pocket'), screen(X,yes); screen(X,no).


cleaning(X) :- askobj('a liquid'), cleaning_liquid(X); cleaning_device(X).
cleaning_liquid(X) :- askfunction('clean other objects'), liquid(X,yes); liquid(X,no).

students(X) :- askfunction('contain other objects'), container(X).
students(X) :- askfunction('open things'), opener(X).

container(X) :- askfunction('fit in a pocket'), contain(X,yes); contain(X,no).

kitchen_obj(X) :- askobj('used for cooking'), cookware(X).
kitchen_obj(X) :- askobj('used for eating'), eating(X).

eating(X) :- askobj('cutlery'), cutlery(X).
eating(X) :- askobj('dishware'), dishware(X).

furnitures(X) :- askobj('used for sleeping'), furniture(X,yes); furniture(X,no).

appliance(X) :- appliance(X,Y), boolean(Y).
appliance(range,yes).
appliance(oven,no).
small_appliance(X) :- small_appliance(X,Y), boolean(Y).
small_appliance(coffee_machine,yes).
small_appliance(toaster,no).

e_cleaning_device(vacuum).
lighting(lamp).

screen(X) :- screen(X,Y), boolean(Y).
screen(phone,yes).
screen(computer,no).


liquid(dishwashing_detergent,yes).
liquid(shampoo,no).
cleaning_device(broom).

contain(X) :- contain(X,Y), boolean(Y).
contain(wallet,yes).
contain(backpack,no).
opener(key).

cookware(pan).
cutlery(fork).
dishware(plate).

furniture(X) :- furniture(X,Y), boolean(Y).
furniture(bed,yes).
furniture(table,no).

instrument(piano).
plant(cactus).
sheet(paper).
