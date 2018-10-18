% Include necessary files
include "globals.mzn";
include "alldifferent.mzn";

% fetch the given data
int: nbTeams;
array [1..nbTeams, 1..nbTeams] of var 0..1: pv;
int: nMatches = nbTeams -1;

% Initialize table of matches that will be played
array [1..nbTeams, 1..nMatches] of var 1..nbTeams: matches;
% each row is a team and each column is a round. The value for (r,c) is the team that team r will be playing against




% Each item in a row is unique
predicate row_diff(int: r) = alldifferent([matches[r, i] | i in 1..nMatches]);

% Each item in a column is unique
predicate col_diff(int: c) = alldifferent ([matches[i, c] | i in 1..nbTeams]);

% A team can not play itself
predicate not_itself(int: r) = forall(c in 1..nMatches) (matches[r,c] != r);



% If team1 plays at home, team2 plays away
%predicate home_team(int: t1, var int: t2) = (pv[t1,t2] == 1 -> pv[t2,t1] == 0);

% If team1 plays away, team2 plays at home
%predicate away_team(int: t1, var int: t2) = (pv[t1,t2] == 0 -> pv[t2,t1] == 1);

% Symmetry of one team playing at home implying the other plays away
predicate home_away_team(int: t1, var int: t2) = (pv[t1,t2] == 1 <-> pv[t2,t1] == 0);







% Unique items in rows
constraint forall (r in 1..nbTeams) (row_diff(r)); % Pas de répétition sur les lignes

% Unique items in columns
constraint forall (c in 1..nMatches) (col_diff(c)); % Pas de répétition sur les colonnes

% Teams can not play themselves
constraint forall (r in 1..nbTeams) (not_itself(r)); % A team does not play itself





% If team1 plays team2, team2 must play team1
%constraint forall(t1 in 1..nbTeams, t2 in 1..nbTeams, m in 1..nMatches)(
%      matches[t1,m] == t2 -> matches[t2,m] == t1
%);

% If team2 plays team1, team1 must play team2
%constraint forall(t1 in 1..nbTeams, t2 in 1..nbTeams, m in 1..nMatches)(
%      matches[t2,m] == t1 -> matches[t1,m] == t2
%);

% Symmetry of teams playing eachother
constraint forall(t1 in 1..nbTeams, t2 in 1..nbTeams, m in 1..nMatches)(
      matches[t1,m] == t2 <-> matches[t2,m] == t1
);




% Ensuring that the constraints given regarding home matches are respected
%constraint forall(t in 1..nbTeams, m in 1..nMatches)(
%    home_team(t, matches[t,m])    
%);

% Ensuring that the constraints given regarding away matches are respected
%constraint forall(t in 1..nbTeams, m in 1..nMatches)(
%    away_team(t, matches[t,m])    
%);

% Ensuring that the constraints given regarding home/away matches are respected
constraint forall(t in 1..nbTeams, m in 1..nMatches)(
    home_away_team(t, matches[t,m])
);






% BONUS: keep rounds to a max of 4 home games in a row
constraint forall(t in 1..nbTeams, m in 1..(nMatches - 3))(
    pv[t, matches[t, (m+0)]] == 0 \/ pv[t, matches[t, (m+1)]] == 0 \/ pv[t, matches[t, (m+2)]] == 0 \/ pv[t, matches[t, (m+3)]] == 0
);



% Solve problem
solve satisfy;


% Output results
output [join("\n", [join("\t",[show(matches[r,m]) | m in 1..nMatches]) | r in 1..nbTeams])];
output ["\n\n"];
output [join("\n", [join("\t",[show(pv[r,matches[r,m]]) | m in 1..nMatches]) | r in 1..nbTeams])];