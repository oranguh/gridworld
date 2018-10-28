import numpy as np
from Coordinate import Coordinate
from random import randint
import operator
import numpy

def main():
	grid_size = 5
	gamma = 0.9
	# strategy = 'greedy'
	strategy = 'qlearn_greedy'
	# strategy = 'equi'
	epsilon = 0.7
	# epsilon = 1

	my_grid = Gridworld(grid_size, gamma, strategy, epsilon)
	# my_grid.simulate_play(num_of_iterations = 1000)
	# my_grid.simulate_play_episodes(episodes = 100)
	my_grid.Qlearning(num_of_iterations = 10000)
	# my_grid.sarsa(num_of_iterations = 1000)

	for key in my_grid.coordinates:
		print ('Coordinate: {}, value:{}, count:{}'.format(key, my_grid.coordinates[key].value, my_grid.coordinates[key].count))

	my_grid.pretty_printings()
	my_grid.pretty_printings_qsa()


class Gridworld:

	def __init__(self, grid_size, discount_factor, strategy, epsilon):
		self.grid_size = grid_size

		# also known as gamma
		self.discount_factor = discount_factor
		self.strategy = strategy

		# exploration for greedification
		self.epsilon = epsilon
		self.coordinates = self.initialize_grid()
		self.agent_position = self.initialize_agent()

		# learning rate
		self.alpha = 1
		self.actions_made = 1

	def initialize_grid(self):
		coordinates = {}
		for position_x in range(self.grid_size):
			for position_y in range(self.grid_size):
				coordinates[(position_x, position_y)] = (Coordinate(position_x, position_y))

		return coordinates


	def initialize_agent(self):
		coordinate_x = randint(0, self.grid_size - 1)
		coordinate_y = randint(0, self.grid_size - 1)

		return (coordinate_x, coordinate_y)


	def move_agent(self):
		instant_reward = 0

		# Calculate probabilities for next move
		p_N, p_E, p_S, p_W = self.calculate_probabilities(self.agent_position)

		# Perform next move
		random_move = self.coordinates[self.agent_position].next_move(p_N, p_E, p_S, p_W)
		next_position = tuple(map(operator.add, self.agent_position, random_move))

		# Check if out of bounds
		while self.check_out_of_bounds(next_position):
			random_move = self.coordinates[self.agent_position].next_move(p_N, p_E, p_S, p_W)
			next_position = tuple(map(operator.add, self.agent_position, random_move))
			instant_reward -= 1

		# Special coordinate A
		if self.agent_position == (4, 1):
			instant_reward = 10
			next_position = (0, 1)

		# Special coordinate B
		if self.agent_position == (4, 3):
			instant_reward = 5
			next_position = (2, 3)


		return instant_reward, next_position, random_move



	def check_out_of_bounds(self, next_position):
		is_out_of_bounds = False
		for position in next_position:
			if position < 0 or position > 4:
				is_out_of_bounds = True

		return is_out_of_bounds


	def calculate_next_possible_move(self, current_coordinate, action):
		# equiprobability approach

		next_possible_reward = 0
		if action == 'N':
			next_possible_position = tuple(map(operator.add, self.agent_position, (0,1) ))
			if self.check_out_of_bounds(next_possible_position):
				next_possible_reward = -1
				next_possible_position = current_coordinate

		if action == 'E':
			next_possible_position = tuple(map(operator.add, self.agent_position, (1,0) ))
			if self.check_out_of_bounds(next_possible_position):
				next_possible_reward = -1
				next_possible_position = current_coordinate

		if action == 'S':
			next_possible_position = tuple(map(operator.add, self.agent_position, (0,-1) ))
			if self.check_out_of_bounds(next_possible_position):
				next_possible_reward = -1
				next_possible_position = current_coordinate

		if action == 'W':
			next_possible_position = tuple(map(operator.add, self.agent_position, (-1,0) ))
			if self.check_out_of_bounds(next_possible_position):
				next_possible_reward = -1
				next_possible_position = current_coordinate

		if current_coordinate == (4, 1):
			next_possible_reward = 10
			next_possible_position = (0, 1)

		if current_coordinate == (4, 3):
			next_possible_reward = 5
			next_possible_position = (2, 3)

		return next_possible_reward, next_possible_position


	def calculate_probabilities(self, current_coordinate):
		if self.strategy == 'equi':
			return 0.25, 0.25, 0.25, 0.25
		elif self.strategy == 'qlearn_greedy':
			actions = ['N', 'E', 'S', 'W']
			moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
			best_move = self.coordinates[current_coordinate].find_max_action()
			#Assign probability to biggest value
			p_N = self.epsilon / 3
			p_E = self.epsilon / 3
			p_S = self.epsilon / 3
			p_W = self.epsilon / 3
			if best_move == (0, 1):
				p_N = 1 - self.epsilon
			elif best_move == (1, 0):
				p_E = 1 - self.epsilon
			elif best_move == (0, -1):
				p_S = 1 - self.epsilon
			elif best_move == (-1, 0):
				p_W = 1 - self.epsilon

			return p_N, p_E, p_S, p_W

		else:
			actions = ['N', 'E', 'S', 'W']
			moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
			max_value = -9999

			#Identify neighbor with biggest value
			for action_index, move in enumerate(moves):
				next_possible_position = tuple(map(operator.add, current_coordinate, move ))
				if self.check_out_of_bounds(next_possible_position):
					next_possible_position = current_coordinate

				if self.coordinates[next_possible_position].value > max_value:
					max_value = self.coordinates[next_possible_position].value
					best_move = actions[action_index]

			#Assign probability to biggest value
			p_N = self.epsilon / 3
			p_E = self.epsilon / 3
			p_S = self.epsilon / 3
			p_W = self.epsilon / 3
			if best_move == 'N':
				p_N = 1 - self.epsilon
			elif best_move == 'E':
				p_E = 1 - self.epsilon
			elif best_move == 'S':
				p_S = 1 - self.epsilon
			elif best_move == 'W':
				p_W = 1 - self.epsilon

			return p_N, p_E, p_S, p_W


	def calculate_value(self, current_coordinate):
		self.coordinates[current_coordinate].count += 1
		reward_N, position_N = self.calculate_next_possible_move(current_coordinate, 'N')
		# print(position_N, reward_N)
		reward_E, position_E = self.calculate_next_possible_move(current_coordinate, 'E')
		# print(position_E, reward_E)
		reward_S, position_S = self.calculate_next_possible_move(current_coordinate, 'S')
		# print(position_S, reward_S)
		reward_W, position_W = self.calculate_next_possible_move(current_coordinate, 'W')
		# print(position_W, reward_W)

		p_N, p_E, p_S, p_W = self.calculate_probabilities(current_coordinate)
		# print(p_N, p_E, p_S, p_W)
		value = p_N*(reward_N + self.discount_factor*self.coordinates[position_N].value) + p_E*(reward_E + self.discount_factor*self.coordinates[position_E].value) + p_S*(reward_S + self.discount_factor*self.coordinates[position_S].value) + p_W*(reward_W + self.discount_factor*self.coordinates[position_W].value)

		return value


	def simulate_play(self, num_of_iterations = 10):
		for iteration in range(num_of_iterations):
			# Get current position
			previous_coordinate = self.agent_position
			# print(previous_coordinate)
			# Update values
			self.coordinates[previous_coordinate].value = self.calculate_value(previous_coordinate)
			# Move agent
			instant_reward, current_coordinate, action_made = self.move_agent()
			# Update agent's coordinate
			self.agent_position = current_coordinate

	def simulate_play_episodes(self, episodes = 10):
		for iteration in range(episodes):
			coordinate_history = []
			coordinate_history.append(self.agent_position)
			# creates a whole episode, which stops only when terminal node is found
			while ((self.agent_position != (4, 1)) and (self.agent_position != (4, 3))):
				# Get current position
				previous_coordinate = self.agent_position
				# Move agent
				instant_reward, current_coordinate, action_made = self.move_agent()
				# Update agent's coordinate
				self.agent_position = current_coordinate
				# Save Coordinate history
				coordinate_history.append(self.agent_position)

			# with this episode, values are calculated for each state (no discounting)
			reward = 0
			for coordinate in reversed(coordinate_history):
				if (coordinate == (4, 1)):
					self.coordinates[coordinate].value = 10
					reward = 10
				elif (coordinate == (4, 3)):
					self.coordinates[coordinate].value = 5
					reward = 5
				else:
					# update value
					old = self.coordinates[coordinate].count * self.coordinates[coordinate].value
					# update count
					self.coordinates[coordinate].count += 1
					new = reward
					self.coordinates[coordinate].value = (old + new)/self.coordinates[coordinate].count
					# update reward if wall
					if self.check_out_of_bounds(coordinate):
						reward -= 1

			self.agent_position = self.initialize_agent()

	def Qlearning(self, num_of_iterations = 100):

		"""" index problems...."""
		for iteration in range(num_of_iterations):

			# Get current position
			previous_coordinate = self.agent_position

			# Move agent
			instant_reward, current_coordinate, action_made = self.move_agent()
			previous_index = [(0, 1), (1, 0), (0, -1), (-1, 0)].index(action_made)
			self.actions_made += 1

			# alpha = 1/t. But is t the 'total' amount of actions made? or the amount of 'specific' actions made.
			self.alpha = 1/self.coordinates[previous_coordinate].actionset[previous_index].count
			# self.alpha = 0.5

			# qsa = qsa + alpha(R + epsilon(max(q's'a')) - qsa)
			# print(previous_index)

			self.coordinates[previous_coordinate].actionset[previous_index].value = \
			(1 - self.alpha) * self.coordinates[previous_coordinate].actionset[previous_index].value + \
			self.alpha * (self.coordinates[previous_coordinate].reward + (self.discount_factor * \
			self.coordinates[self.agent_position].find_max_action_value()))

			# self.coordinates[previous_coordinate].actionset[previous_index].value = \
			# self.coordinates[previous_coordinate].actionset[previous_index].value + \
			# self.alpha * (self.coordinates[previous_coordinate].reward + self.discount_factor * \
			# self.coordinates[self.agent_position].find_max_action_value() - \
			# self.coordinates[previous_coordinate].actionset[previous_index].value)


			# print(self.coordinates[previous_coordinate].actionset[previous_index].value)
			self.coordinates[previous_coordinate].actionset[previous_index].count += 1

			# Update agent's coordinate
			self.agent_position = current_coordinate

	def sarsa(self, num_of_iterations = 100):
		pass

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
		for position_x in reversed(range(self.grid_size)):
			for position_y in range(self.grid_size):
				print("|  ", end = '')
				for i, action in enumerate(["N", "E", "S", "W"]):
					print(action, end = '')
					print(":{0:.2f}  ".format(self.coordinates[(position_x, position_y)].actionset[i].value), end = '')
				print("  ", end = '')
			print("|")
			print()
			print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
			print()



if __name__ == "__main__":
    main()
