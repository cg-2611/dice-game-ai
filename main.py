import numpy as np
import sys
import time

from agent import dice_game_agent
from game import dice_game

def get_flags():
    """
    Parse the command line arguments and search for any specified options.

    :return: n, seed, verbose
             n: the number of games to be played by the agent, or a default of 10 if option not provided
             seed: the seed to be used by the random number generator, or None if option not provided
             verbose: true if the option is specified, false otherwise
    """
    n_index = sys.argv.index("-n") if "-n" in sys.argv else -1
    n = int(sys.argv[n_index + 1]) if n_index != -1 else 10

    seed_index = sys.argv.index("--seed") if "--seed" in sys.argv else -1
    seed = int(sys.argv[seed_index + 1]) if seed_index != -1 else None

    verbose = True if "--verbose" in sys.argv else False

    return n, seed, verbose

def get_dice_flags():
    """
    Parse the command line arguments and search for any specified options for the dice.
    If an option is not specified, None is returned so that the default is used in the DiceGame constructor.

    :return: dice, sides, values, biases, penalty
             dice: the number of dice to be used
             sides: the number of sides of each dice has
             values: a list of values of the dice
             biases: a list of probabilities for each value
             penalty: the penalty for re-rolling the dice
    """
    dice_index = sys.argv.index("-d") if "-d" in sys.argv else -1
    next = sys.argv[dice_index + 1]
    dice = int(next) if dice_index != -1 and not next.startswith("-") else None

    sides_index = sys.argv.index("-s") if "-s" in sys.argv else -1
    next = sys.argv[sides_index + 1]
    sides = int(next) if sides_index != -1 and not next.startswith("-") else None

    values_index = sys.argv.index("-v") if "-v" in sys.argv else -1
    next = sys.argv[values_index + 1]
    values = [int(value) for value in next.split(",")] if values_index != -1 and not next.startswith("-") else None

    biases_index = sys.argv.index("-b") if "-b" in sys.argv else -1
    next = sys.argv[biases_index + 1]
    biases = [float(bias) for bias in next.split(",")] if biases_index != -1 and not next.startswith("-") else None

    penalty_index = sys.argv.index("-p") if "-p" in sys.argv else -1
    next = sys.argv[penalty_index + 1]
    penalty = int(next) if penalty_index != -1 and not next.startswith("-") else None

    return dice, sides, values, biases, penalty

def play(game: dice_game.DiceGame, agent: dice_game_agent.DiceGameAgent, verbose:bool) -> int:
    """
    Plays the game using the agent and accumulates a score to return.

    :param game: the DiceGame instance to be played
    :param agent: the DiceGameAgent playing the game
    :param verbose: control the output to show the agents action for each dice roll
    :return: the score for the agent after playing the game
    """
    state = game.reset()

    if verbose:
        print(f"Initial dice: {state}\n")

    game_over = False
    roll_count = 0

    while not game_over:
        roll_count += 1

        action = agent.get_action(state)
        state, game_over = game.roll(action)

        if verbose:
            print(f"Agent action: \t{action}")

            if not game_over:
                print(f"Dice roll {roll_count}: \t{state}")


    if verbose:
        print(f"\nFinal dice: {state}")

    return game.score

def main() -> None:
    total_games, seed, verbose = get_flags()
    np.random.seed(seed)

    dice, sides, values, biases, penalty = get_dice_flags()
    game = dice_game.DiceGame(dice, sides, values=values, biases=biases, penalty=penalty)

    policy_iteration_start = time.process_time()
    agent = dice_game_agent.DiceGameAgent(game)
    policy_iteration_end = time.process_time()

    print()

    total_score = 0
    for i in range(1, total_games + 1):
        if verbose:
            print(f"Game {i}:")

        score = play(game, agent, verbose)
        total_score += score

        print(f"Game {i} score: {score}")

        if verbose:
            print("----------------------")

    print()
    print(f"Time to find optimal policy: {policy_iteration_end - policy_iteration_start:.4f}s")
    print(f"Average score over {total_games} games: {total_score / total_games}")

if __name__ == "__main__":
    main()
