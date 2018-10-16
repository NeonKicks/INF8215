include "globals.mzn";
include "alldifferent.mzn";

% Number of people/houses/drinks...
int: N = 5;


% set of string Categories: {"Drinks", "Colors", "Nationalities", "Professions", "Animals"}
enum Drinks = {coffee, milk, tea, orangeJuice, water};
enum Colors = {red, green, blue, yellow, white};
enum Nationalities = {ukranian, english, spanish, japanese, norwegian};
enum Professions = {physician, diplomat, sculpter, acrobat, violinist};
enum Animals = {dog, snails, fox, horse, zebra};
set of int: Positions = 1..N;




array[1..N] of var Drinks: homeDrinks;
array[1..N] of var Colors: homeColors;
array[1..N] of var Nationalities: homeNationalities;
array[1..N] of var Professions: homeProfessions;
array[1..N] of var Animals: homeAnimals;
array[1..N] of var Positions: homePositions;


% Ensure that all values are different

constraint alldifferent(homeDrinks);
constraint alldifferent(homeColors);
constraint alldifferent(homeNationalities);
constraint alldifferent(homeProfessions);
constraint alldifferent(homeAnimals);
constraint alldifferent(homePositions);








% 1)
constraint forall(i in 1..N)(
  homeNationalities[i] == english <-> homeColors[i] == red );
  
% 2)
constraint forall(i in 1..N)(
  homeNationalities[i] == spanish <-> homeAnimals[i] == dog );
  
% 3)
constraint forall(i in 1..N)(
  homeColors[i] == green <-> homeDrinks[i] == coffee );
  
% 4)
constraint forall(i in 1..N)(
  homeNationalities[i] == ukranian <-> homeDrinks[i] == tea );
  
% 5)
constraint forall(i,j in 1..N)(  
  homeColors[i] == green /\ homeColors[j] == white -> homePositions[i] == homePositions[j] + 1 );
  
% 6)
constraint forall(i in 1..N)(
  homeProfessions[i] == sculpter <-> homeAnimals[i] == snails );
  
% 7)
constraint forall(i in 1..N)(
  homeProfessions[i] == diplomat <-> homeColors[i] == yellow );
  
% 8)
constraint forall(i in 1..N)(
  homeDrinks[i] == milk <-> homePositions[i] == 3 );  

% 9)
constraint forall(i in 1..N)(
  homeNationalities[i] == norwegian <->homePositions[i] == 1 );

% 10)
constraint forall(i,j in 1..N)(
  homeProfessions[i] == physician /\ homeAnimals[j] == fox -> homePositions[i] - homePositions[j] == 1 \/ homePositions[i] - homePositions[j] == -1 );
  
% 11)
constraint forall(i,j in 1..N)(
  homeProfessions[i] == diplomat /\ homeAnimals[j] == horse -> homePositions[i] - homePositions[j] == 1 \/ homePositions[i] - homePositions[j] == -1 ); 

% 12)
constraint forall(i in 1..N)(
  homeProfessions[i] == violinist <-> homeDrinks[i] == orangeJuice );
  
% 13)
constraint forall(i in 1..N)(
  homeNationalities[i] == japanese <-> homeProfessions[i] == acrobat );

% 14)
constraint forall(i,j in 1..N)(
  homeNationalities[i] == norwegian /\ homeColors[j] == blue -> homePositions[i] - homePositions[j] == 1 \/ homePositions[i] - homePositions[j] == -1 );








solve satisfy;


output [join("\n",[join("\t",[show(homePositions[i]), show(homeNationalities[i]), show(homeProfessions[i]), show(homeAnimals[i]), show(homeColors[i]), show(homeDrinks[i])]) | i in 1..N])];


