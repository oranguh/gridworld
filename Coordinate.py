import numpy
import random
from scipy import stats
import numpy as np


class Coordinate:

    def __init__(self, coordinate_x, coordinate_y):
        self.coordinate_x = coordinate_x
        self.coordinate_y = coordinate_y
        self.count = 0
        self.value = 0
        self.reward = -1
        self.terminal = False

        self.actionset = []

        for action in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            self.actionset.append(
                Action(self.coordinate_x, self.coordinate_y, action))

    def next_move(self, p_N, p_E, p_S, p_W):
        north = (0, 1)
        east = (1, 0)
        south = (0, -1)
        west = (-1, 0)

        xk = np.arange(4)
        pk = (p_N, p_E, p_S, p_W)
        action_distribution = stats.rv_discrete(
            name='action_distribution', values=(xk, pk))
        move_int = action_distribution.rvs()

        if move_int == 0:
            return north
        elif move_int == 1:
            return east
        elif move_int == 2:
            return south
        elif move_int == 3:
            return west

    def find_max_action(self):
        """ returns max valued action, returns random action if equal value"""

        best_actions = []
        action_values = []

        for action in self.actionset:
            action_values.append(action.value)

        for action in self.actionset:
            if (action.value == max(action_values)):
                best_actions.append(action.action)

        return random.choice(best_actions)

    def find_max_action_value(self):
        """ returns max valued action, returns random action if equal value"""

        best_actions = []
        action_values = []

        for action in self.actionset:
            action_values.append(action.value)
        # print(action_values)
        # print(max(action_values))
        for action in self.actionset:
            if (action.value == max(action_values)):
                best_actions.append(action.action)

        return max(action_values)

    def show_actionset(self):
        print("The coordinate {} {} contains the following actions: ".format(
            self.coordinate_x, self.coordinate_y))

        for action in self.actionset:
            print("Moving {} has a value of {} and has been chosen {} times".format(
                action.action, action.value, action.count))


class Action:
    def __init__(self, coordinate_x, coordinate_y, action):
        self.coordinate_x_state = coordinate_x
        self.coordinate_y_state = coordinate_y
        self.action = action
        self.count = 1
        self.value = 0
