import functools
import itertools
import numpy as np

from scipy import stats


class DiceGame:
    def __init__(self, dice=None, sides=None, *, values=None, biases=None, penalty=None) -> None:
        """
        Constructor for DiceGame, creates a new instance of a DiceGame class.

        :param dice: the number of dice in the game, 3 is the default
        :param sides: the number of sides each dice has, 6 is the default
        :param values: a list of values that the dice can have, 1 to number of sides is default
        :param biases: a list of probabilities that correspond the value at the same index of the values list,
                       the probability controls how often a given value will be rolled,
                       equal probability for each value is default
        :param penalty: the cost of re-rolling the dice that is subtracted from the score each time the
                        the dice are rolled and the game is not over, default is 1
        """
        self.__dice = dice if dice is not None else 3
        self.__sides = sides if sides is not None else 6
        self.__values =  np.array(values) if values is not None else np.arange(1, self.__sides + 1)
        self.__biases =  np.array(biases) if biases is not None else (np.ones(self.__sides) / self.__sides)
        self.__penalty = penalty if penalty is not None else 1

        if len(self.__values) != self.__sides:
            raise ValueError("values must have same length as sides")

        if len(self.__values) != len(self.__biases):
            raise ValueError("biases and values must be same length")

        self.__opposite_sides = {side: opposite for side, opposite in zip(self.__values, self.__values[::-1])}

        self.states = [state for state in itertools.combinations_with_replacement(self.__values, self.__dice)]

        self.actions = [()]
        for i in range(1, self.__dice + 1):
            self.actions.extend(itertools.combinations(range(self.__dice), i))

        self.reset()

    def __final_score(self, dice_values: tuple) -> np.ndarray:
        # get a list of unique values from the dice values and the count of each one in another list
        unique_values, counts = np.unique(dice_values, return_counts=True)

        # for any unique values with a count greater than one, use the value of the opposite side of the dice
        unique_values[counts > 1] = np.array([self.__opposite_sides[x] for x in unique_values[counts > 1]])
        return np.sum(unique_values[counts == 1]) + np.sum(unique_values[counts > 1] * counts[counts > 1])

    def __flip_duplicates(self) -> None:
        unique_values, counts = np.unique(self.__current_dice, return_counts=True)
        if np.any(counts > 1):
            # flip the values of the current dice to the value of the opposite side if there are any counts > 1
            # mask is an ndarray of boolean values that represent if the value from the current dice is in the
            # unique values list with a count > 1
            mask = np.isin(self.__current_dice, unique_values[counts > 1])
            self.__current_dice[mask] = [self.__opposite_sides[x] for x in self.__current_dice[mask]]

        self.__current_dice.sort()

    def reset(self):
        """
        Reset the game.

        :return: a tuple of new dice values
        """
        self.__game_over = False
        self.score = self.__penalty
        self.__current_dice = np.zeros(self.__dice, dtype=np.intc)
        dice, _ = self.roll()
        return dice

    def get_dice_state(self) -> tuple:
        """
        Get the current state of the dice in the game.

        :return: a tuple of the current rolled dice values
        """
        return tuple(self.__current_dice)

    def roll(self, action=()):
        """
        Re-rolls the dice that where not held.
        :param action: a tuple of indexes that correspond to which dice are to be held (not re-rolled)
        :return: state, game_over
                 state:
                    a tuple containing the new dice values after applying the action to the current state
                    and replacing the non-held dice values with new, random values from the available values
                 game_over:
                    has value true if the action taken is to hold all dice, indicating the game is over
        """
        if action not in self.actions:
            raise ValueError(f"{action} not a valid action for current game")

        if self.__game_over:
            return 0

        hold_count = len(action)

        if hold_count == self.__dice:
            # if the game is over (all dice held), calculate return the final dice score, the final dice values
            # and True to indicate the game is over
            self.__flip_duplicates()
            self.score += np.sum(self.__current_dice)
            return self.get_dice_state(), True
        else:
            # if the game is not over
            # set the mask element at the index of the dice that were held to false
            mask = np.ones(self.__dice, dtype=np.bool_)
            action = np.array(action, dtype=np.intc)
            mask[action] = False

            # replace the values of the dice not held with a new random value from the possible values
            not_held = self.__dice - hold_count
            self.__current_dice[mask] = np.random.choice(self.__values, size= not_held, replace=True, p=self.__biases)

            self.__current_dice.sort()
            self.score -= self.__penalty

            return self.get_dice_state(), False

    def get_next_states(self, state: tuple, action: tuple):
        """
        Get all possible resulting states from taking a given action in a given state.

        :param state: the current state
        :param action: the action taken from the current state
        :return: possible_states, game_over, reward, probabilities
                 possible_states:
                    a list containing each possible resulting state as tuples,
                    if the game is over then a list containing None is returned
                 game_over:
                    has value true if the action taken is to hold all dice, indicating the game is over
                 reward:
                    the reward for this action, if the game is not over, -1 * penalty is returned,
                    if the game is over, then the final value of the dice is returned
                 probabilities:
                    a list of probabilities, each corresponding to a state in possible_states, representing the
                    probability of reaching each state after taken the given action from the current state
        """
        if state not in self.states:
            raise ValueError(f"{state} not a valid state for current game")

        if action not in self.actions:
            raise ValueError(f"{action} not a valid action for current game")

        hold_count = len(action)

        if hold_count == self.__dice:
            return [None], True, self.__final_score(state), np.array([1])
        else:
            # mask contains true when a value should be held and false when not
            mask = np.zeros(self.__dice, dtype=np.bool_)
            hold = np.array(action, dtype=np.intc)
            mask[hold] = True

            # get all possible combinations of states for the values not held
            state_combinations = itertools.combinations_with_replacement(self.__values, self.__dice - hold_count)
            possible_states = np.array(list(state_combinations), dtype=np.intc)

            states_index = itertools.combinations_with_replacement(range(self.__sides), self.__dice - hold_count)
            possible_states_index = np.array(list(states_index), dtype=np.intc)

            # calculate the probability of each possible state using a multinomial distribution
            partial = functools.partial(np.bincount, minlength=self.__sides)
            queries = np.apply_along_axis(partial, 1, possible_states_index)
            probabilities = stats.multinomial.pmf(queries, self.__dice - hold_count, self.__biases)

            state_as_array = np.asarray(state, dtype=np.intc)
            possible_states = np.insert(possible_states, np.zeros(hold_count, dtype=np.intc), state_as_array[mask], axis=1)

            possible_states = np.sort(possible_states, axis=1)
            possible_states = [tuple(state) for state in possible_states]

            return possible_states, False, (-1 * self.__penalty), probabilities
