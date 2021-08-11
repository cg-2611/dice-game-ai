class DiceGameAgent:
    def __init__(self, game) -> None:
        """
        Constructor for DiceGameAgent, creates a new instance of a DiceGameAgent class.

        :param game: the DiceGame instance that the agent will play
        """
        self.game = game

        # these values were chosen through trial and error
        self.__theta = 0.001 # controls when policy iteration decides convergence has occurred
        self.__gamma = 0.94  # the discount rate used in the Bellman equation

        # dictionary that contains every state and the optimal actin to take at each state
        self.__optimal_policy = self.__policy_iteration()

    def __policy_iteration(self) -> dict:
        i = len(self.game.actions) - 1 # index that will be used to loop through the game actions

        # state-value function values, give each state an initial value of 0
        v = {state: 0 for state in self.game.states}

        # initially make each action for each state the hold action
        policy = {state: self.game.actions[i] for state in self.game.states}

        # while the policy has not yet converged, evaluate and improve the policy
        policy_stable = False
        while not policy_stable:
            v = self.__policy_evaluation(v, policy)
            i, policy_stable, policy = self.__policy_improvement(i, v, policy)

        return policy # this is the optimal policy for self.game

    def __policy_evaluation(self, v: dict, policy: dict) -> None:
        converged = False

        while not converged:
            delta = 0
            for state in self.game.states:
                old_v = v[state]
                new_v = 0
                action = policy[state]

                new_states, game_over, reward, probabilities = self.game.get_next_states(state, action)

                # compute new value of the state using the Bellman equation
                for i in range(len(new_states)):
                    if not game_over:
                        new_v += probabilities[i] * (reward + (self.__gamma * v[new_states[i]]))
                    else:
                        new_v += (reward + (self.__gamma * v[state]))

                # update the state-value function dictionary with the new value if it is an improvement
                if new_v > v[state]:
                    v[state] = new_v

                # stop iterating if the difference between the old and new value
                # is less than the threshold for convergence
                delta = max(delta, abs(old_v - v[state]))
                if delta < self.__theta:
                    converged = True

        return v # updated state-value function values using the improved policy

    def __policy_improvement(self, action_index: int, v: dict, policy: dict) -> None:
        stable = True
        new_policy = policy

        # decremented to interate backwards through list of actions each time function called
        action_index -= 1

        for state in self.game.states:
            old_action = policy[state]
            value = 0

            # new_action will be assigned to any states in the policy where it improves the states value
            new_action = self.game.actions[action_index]

            new_states, game_over, reward, probabilities = self.game.get_next_states(state, new_action)

            # compute the value of a state using the Bellman equation
            for i in range(len(new_states)):
                if not game_over:
                    value += probabilities[i] * (reward + (self.__gamma * v[new_states[i]]))
                else:
                    value += (reward + (self.__gamma * v[state]))

            # replace the action taken from the state with if this action maximises the value of Bellman equation so far
            if value > v[state]:
                new_policy[state] = new_action

            # if policy has changed in any way, then the policy iteration is incomplete and must continue
            # otherwise it can terminate
            if old_action != new_policy[state]:
                stable = False

        return action_index, stable, new_policy

    def get_action(self, state: tuple) -> tuple:
        """
        Get the optimal action to take at the given state.

        :return: the optimal action for current state
        """
        return self.__optimal_policy[state]
