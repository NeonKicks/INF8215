include "globals.mzn";
include "alldifferent.mzn";

% fetch the given data
int: nbTeams;
array [1..nbTeams, 1..nbTeams] of var 0..1: pv;

