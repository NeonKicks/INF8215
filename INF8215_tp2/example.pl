% example file

human(scott).
human(ramona).
human(envy).
human(wallace).
girl(ramona).
enemies(scott, envy).
enemies(X, Y) :- enemies(Y, X).