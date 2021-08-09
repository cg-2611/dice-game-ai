# Dice Game AI

This was a coursework assignment I was given during my first year at university. It is an artificial intelligence agent that uses policy iteration to produce an optimal policy for a game involving dice. 

I did not produce the code for the game itself, that was given to me. This repository contains a version of the game class located in `game/dice_game.py` that has some minor modifications by myself (only for improved readability, no modifications to the logic of the program). Most of the logic and mathematical calculations are not my own and credit goes to my university.


### Contents:
- [The Game](#the-game)
- [Policy Iteration](#policy-iteration)
- [Run](#run)
- [Options](#options)

### The Game
---
The dice game involves a specified number of dice, each with a specified number of sides. 

The dice are rolled, and the values of each dice are added together to produce a score. If any duplicates are rolled (i.e. multiple of the same value), before the final score is calculated, the duplicate values are flipped, and the corresponding number on the opposite side of the dice is used instead for all occurrences of the duplicate value. 

The player can choose to hold any combination of the dice before re-rolling (any dice held are not re-rolled). Every time the dice are re-rolled, a specified penalty is deducted from the final score.

The aim of the game is to score as high as possible.


### Policy Iteration
---
Policy iteration is an algorithm used to produce a solution for a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) problem. By using policy iteration, an optimal policy for a state-space can be determined. It does this by repeatedly evaluating the current best policy using the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation), and then making improvements to it until the policy converges, i.e. there are no more improvements to be made, at this point the optimal policy has been found. The result of policy iteration is a policy that contains the optimal action to take from a given state for every state in a problems state-space.


### Run
---
First clone the repository with:
```
git clone https://github.com/cg-2611/dice-game-ai.git
```
Next, open the directory created by the `git clone` command:
```
cd dice-game-ai
```
The to run the program with default options, run:
```
python main.py
```
> Note: the program requires python 3.9  and so the interpreter used to execute the program must support this version at least. The command `python3` might need to be used instead.


### Options
---
> Note: all the options have defaults within the program and so any combination of options, all options or no options can be supplied to the program.

When executing the program, there are some command line options that I have made available to control how the game is run, the output and the properties of the game instance itself:
- `-b`: the probability biases for the values, this must be provided as a list of floating point numbers the same length as the number of values separated only by a comma and the sum of the probabilities must sum to 1 (default is equal weightings for all values)
- `-d`: the number of dice to be used in the game (default is 3)
- `-n`: the number of games the agent will play (default is 10)
- `-p`: the penalty that is subtracted after every re-roll of the dice (default is 1)
- `-s`: the number of sides of each dice (default is 6)
- `--seed`: the seed for generating random numbers (default is None)
- `-v`: the values of each side of the dice, this must be provided as a list of integers the same length as the number of sides separated only by a comma (default integers from 1 to the number of sides)
- `--verbose`: if used, a flag is set to true and the program will output every action the agent takes during every game, which is used to see what decisions the agent makes (default if not present makes the flag false)

An example of using each option:
```
python main.py -d 6 -s 4 -v 2,4,6,8 -b 0.2,0.5,0.05,0.25 -p 2 -n 100 --seed 123 --verbose
```
This will create a dice game with 6 dice, each with 4 sides of values 2, 4, 6 and 8, side probability weightings of 20% for value 2, 50% for value 4, 5% for value 6 and 25% for value 8 and a penalty of -2 for each re-roll. The agent will play 100 games, with 123 used as the seed for the random numbers and the action taken during each game being output.
