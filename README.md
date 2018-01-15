# Podręcznik użytkownika
## Preprocesing
W katalogu **`./corpus/`** znajdują się przykładowe zbiory tekstów dla czterech języków (holenderski, angielski, hiszpański, grecki). Zawartość jednego z nich należy wkleić do pustego katalogu **`./data/authors/`**, ale plik **`answers.txt`** trzeba umieścić w katalogu głównym projektu. Następnie uruchomić skrypt przygotujący dane binarne dla odpowiedniego języka (dostępne opcje zostaną wyświetlone po nieudanej próbie):

`./data/preprocessing.sh opcja`

W katalogach **`./data/unknown/`** oraz **`./data/known/`** wygenerowane zostaną dane binarne używane przez pozostałe skrypty.

## Trening
Aby rozpocząć trening sieci należy uruchomić skrypt:

`th ./train.lua -h`

Wyświetli on wszystkie modyfikowalne hiperparametry sieci RNN oraz parametry uczenia. Następnie można uruchomić skrypt jeszcze raz z odpowiednio dobranymi parametrami, np.:

`th ./train.lua -max_authors 100 -hidden_size 200 -depth 3 -unroll_times 50 -learning_rate 0.001`

Po przetworzeniu określonej liczbie epok stan RNN będzie persystowany w katalogu **`./snapshots/`**.

## Ewaluacja
Aby zewaluować wytrenowaną sieć, należy zanotować jej **numer** (bez rozszerzenia) z katalogu **`./snapshots/`**, następnie uruchomić skrypt testowy:

`th ./test.lua numer answers.txt`

Po przetworzeniu wszystkich nieznanych tekstów wypisany zostanie średni wynik poprawnych weryfikacji autorstwa.
