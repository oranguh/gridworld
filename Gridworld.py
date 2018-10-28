import numpy as np
from Coordinate import Coordinate
from random import randint
import operator
import numpy
import copy


def main():
    # grid dimensions
    grid_dims = [5, 5]

    # example of how to create special nodes.
    special_nodes = [{"x": 4, "y": 1, "reward": 10, "terminal": True}, {"x": 4, "y": 3, "reward": 5, "terminal": True}]
    # special_node = {"x": 4, "y": 1, "reward": 10}
    # special_nodes.append(special_node)

    simple_grid = Gridworld(grid_dims, special_nodes)

    # print(simple_grid.grid[1, 2].reward)

    simple_grid.Qlearning()
    simple_grid.pretty_printings_qsa()

class Gridworld:
    def __init__(self, grid_dimensions, special_nodes):

        self.grid_dims = grid_dimensions
        # initialize empty grid
        self.grid = np.empty(grid_dimensions, dtype="object")
        # fill grid
        for i, row in enumerate(self.grid):
            for j in range(len(row)):
                self.grid[i, j] = Coordinate(i, j)

        # fill in special nodes
        for special in special_nodes:
            x = special["x"]
            y = special["y"]
            self.grid[x, y].reward = special["reward"]
            self.grid[x, y].terminal = special["terminal"]
        # initialize agent position
        self.agent = \
            (randint(0, self.grid_dims[0] - 1),
             randint(0, self.grid_dims[1] - 1))
        # make action based strategies

    def Qlearning(self, discount_factor=0.9, epsilon_greedy=0.7, iterations=5000):

        for iteration in range(iterations):

            previous_coordinate = copy.deepcopy(self.agent)
            # move agent
            next_position, next_move, been_out_of_bounds = self.qmove_agent(epsilon_greedy)
            self.agent = copy.deepcopy(next_position)
            move_index = [(0, 1), (1, 0), (0, -1),
                          (-1, 0)].index(next_move)
            # get alpha
            alpha = 1 / \
                self.grid[previous_coordinate[0],previous_coordinate[1]].actionset[move_index].count

            # update values
            # if moved out of bounds
            if (been_out_of_bounds):
                print(previous_coordinate, next_position)
                self.grid[previous_coordinate[0],previous_coordinate[1]].actionset[move_index].value = -1

            # if enter a terminal node
            elif (self.grid[previous_coordinate[0],previous_coordinate[1]].terminal == True):
                self.grid[previous_coordinate[0],previous_coordinate[1]].actionset[move_index].value = \
                    (1 - alpha) * self.grid[previous_coordinate[0],previous_coordinate[1]].actionset[move_index].value + \
                    alpha * (self.grid[previous_coordinate[0],previous_coordinate[1]].reward + (discount_factor * \
                    self.grid[self.agent].find_max_action_value()))

                self.agent = \
                    (randint(0, self.grid_dims[0] - 1),
                     randint(0, self.grid_dims[1] - 1))
                continue
            # else normal update
            else:
                self.grid[previous_coordinate[0],previous_coordinate[1]].actionset[move_index].value = \
                    (1 - alpha) * self.grid[previous_coordinate[0],previous_coordinate[1]].actionset[move_index].value + \
                    alpha * (self.grid[self.agent].reward + (discount_factor * \
                    self.grid[self.agent].find_max_action_value()))

    def SARSA(self):
        pass

    def qmove_agent(self, epsilon_greedy):
        been_out_of_bounds = False
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        best_move = self.grid[self.agent[0], self.agent[1]].find_max_action()
        p_N = ((1 - epsilon_greedy)/ 3)
        p_E = ((1 - epsilon_greedy)/ 3)
        p_S = ((1 - epsilon_greedy)/ 3)
        p_W = ((1 - epsilon_greedy)/ 3)

        if best_move == (0, 1):
            p_N = epsilon_greedy
        elif best_move == (1, 0):
            p_E = epsilon_greedy
        elif best_move == (0, -1):
            p_S = epsilon_greedy
        elif best_move == (-1, 0):
            p_W = epsilon_greedy

        next_move = self.grid[self.agent[0],self.agent[1]].next_move(
            p_N, p_E, p_S, p_W)
        next_position = tuple(
            map(operator.add, self.agent, next_move))

        # Check if out of bounds
        # while (self.check_out_of_bounds(next_position)):
        #     been_out_of_bounds = True
        #     next_move = self.grid[self.agent[0],self.agent[1]].next_move(
        #         p_N, p_E, p_S, p_W)
        #     next_position = tuple(
        #         map(operator.add, self.agent, next_move))
        if (self.check_out_of_bounds(next_position)):
            been_out_of_bounds = True
            return self.agent, next_move, been_out_of_bounds


        return next_position, next_move, been_out_of_bounds

    def check_out_of_bounds(self, next_position):
        is_out_of_bounds = False
        for position in next_position:
            if (position < 0 or position > 4):
                is_out_of_bounds = True

        return is_out_of_bounds


    def pretty_printings(self):

        count = 0
        print("----------------------------------------------------------------------------------")
        for position_x in reversed(range(self.grid_size)):
            for position_y in range(self.grid_size):
                print("|\t{0:.2f}\t".format(self.coordinates[(position_x, position_y)].value), end = '')
                count += 1
            print("|")
            print()
            print("----------------------------------------------------------------------------------")
            print()

    def pretty_printings_qsa(self):


        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        for position_x in reversed(range(self.grid_dims[0])):
            for position_y in range(self.grid_dims[1]):
                print("|  ", end = '')
                # for i, action in enumerate(["N", "E", "S", "W"]):
                for i, action in enumerate(["E", "N", "W", "S"]):
                    print(action, end = '')
                    print(":{0:.2f}  ".format(self.grid[position_x, position_y].actionset[i].value), end = '')
                print("  ", end = '')
            print("|")
            print()
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print()



if __name__ == "__main__":
    main()
