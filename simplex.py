import itertools
import numpy
import json

from graph import Graph


class Simplex:
	@staticmethod
	def static_simplex_keys_lazy_but_correct(dimension, size):
		assert dimension >= 0
		assert size >= 0

		for key in itertools.product(range(size + 1), repeat=dimension + 1):
			if sum(key) == size:
				yield numpy.array(key, dtype=numpy.dtype(int))

	@staticmethod
	def static_simplex_keys(dimension, size):
		assert dimension >= 0
		assert size >= 0

		simplex_key = numpy.zeros(dimension + 1, dtype=numpy.dtype(int))
		simplex_key[dimension] = size

		while simplex_key is not None:
			yield simplex_key.copy()

			for i in range(dimension - 1, -1, -1):
				if sum(simplex_key[:i + 1]) < size:
					simplex_key[dimension] -= 1
					simplex_key[i] += 1
					break
				else:
					simplex_key[dimension] += simplex_key[i]
					simplex_key[i] = 0

			if simplex_key[dimension] == size:
				break

	@staticmethod
	def static_unit_ball_key_diffs(dimension):
		assert dimension >= 0

		for i in range(dimension + 1):
			for j in range(dimension + 1):
				if i != j:
					key_diff = numpy.zeros(dimension + 1, dtype=numpy.dtype(int))
					key_diff[i] = +1
					key_diff[j] = -1
					yield key_diff

	@staticmethod
	def static_unit_ball_positive_key_diffs(dimension):
		assert dimension >= 0

		for key_diff in Simplex.static_unit_ball_key_diffs(dimension):
			if numpy.where(key_diff == +1)[0] < numpy.where(key_diff == -1)[0]:
				yield key_diff

	@staticmethod
	def static_unit_vectors_3d():
		return [
			numpy.array([0., 0., 0.]),
			numpy.array([1., 0., 0.]),
			numpy.array([1. / 2., numpy.sqrt(3.) / 2., 0.]),
			numpy.array([1. / 2., 1. / (2. * numpy.sqrt(3.)), numpy.sqrt(2.) / numpy.sqrt(3.)])
		]

	@staticmethod
	def static_key_to_coordinates(key):
		assert len(key) <= 4
		unit_vectors = Simplex.static_unit_vectors_3d()
		coordinates = numpy.zeros(3)
		for i in range(len(key)):
			coordinates += key[i] * unit_vectors[i]
		return coordinates[:len(key) - 1]

	@staticmethod
	def static_compute_simplex_grid_lines(dimension, size):
		assert dimension in [2, 3]

		grid_lines = []

		for simplex_key in Simplex.static_simplex_keys(dimension, size):
			for unit_ball_positive_key_diff in Simplex.static_unit_ball_positive_key_diffs(dimension):
				if all(0 <= x <= size for x in (simplex_key - unit_ball_positive_key_diff)):
					continue

				if not all(0 <= x <= size for x in (simplex_key + unit_ball_positive_key_diff)):
					continue

				i = numpy.where(unit_ball_positive_key_diff == +1)[0][0]
				j = numpy.where(unit_ball_positive_key_diff == -1)[0][0]

				if size - simplex_key[i] <= simplex_key[j] - 0:
					other_key = simplex_key + (size - simplex_key[i]) * unit_ball_positive_key_diff
				else:
					assert size - simplex_key[i] > simplex_key[j] - 0
					other_key = simplex_key + (simplex_key[j] - 0) * unit_ball_positive_key_diff

				assert not numpy.array_equal(simplex_key, other_key)

				# print(simplex_key, other_key)

				grid_lines.append(
					[
						Simplex.static_key_to_coordinates(simplex_key),
						Simplex.static_key_to_coordinates(other_key)
					]
				)

		return grid_lines

	def __init__(self, dimension, size):
		self.dimension = dimension
		self.size = size

		self.simplex_graph = Graph()
		self.dual_graph = Graph()

		self.create_simplex_graph()
		self.create_dual_graph()

	def is_valid_simplex_key(self, key):
		return all(0 <= x <= self.size for x in key) and sum(key) == self.size

	def simplex_keys_lazy_but_correct(self):
		return Simplex.static_simplex_keys_lazy_but_correct(self.dimension, self.size)

	def simplex_keys(self):
		return Simplex.static_simplex_keys(self.dimension, self.size)

	def unit_ball_key_diffs(self):
		return Simplex.static_unit_ball_key_diffs(self.dimension)

	def unit_ball_positive_key_diffs(self):
		return Simplex.static_unit_ball_positive_key_diffs(self.dimension)

	def compute_simplex_grid_lines(self):
		return Simplex.static_compute_simplex_grid_lines(self.dimension, self.size)

	def create_simplex_graph(self):
		for key in self.simplex_keys():
			self.simplex_graph.create_node(key)

		for key in self.simplex_keys():
			for positive_key_diff in self.unit_ball_positive_key_diffs():
				neighbour_key = key + positive_key_diff
				if self.is_valid_simplex_key(neighbour_key):
					node_1 = self.simplex_graph.node(key)
					node_2 = self.simplex_graph.node(neighbour_key)
					self.simplex_graph.create_edge(node_1, node_2, 1.0)

	def create_dual_graph(self):
		for vertex_index in range(self.dimension + 1):
			terminal_dual_key = numpy.zeros(self.dimension + 1)
			for i in range(self.dimension + 1):
				if i == vertex_index:
					terminal_dual_key[i] = -1 * self.size
				else:
					terminal_dual_key[i] = 2 * self.size / self.dimension
			self.dual_graph.create_node(terminal_dual_key)

		for dual_key in Simplex.static_simplex_keys(self.dimension, self.size - 1):
			self.dual_graph.create_node(dual_key)

		for dual_key in Simplex.static_simplex_keys(self.dimension, self.size - 2):
			self.dual_graph.create_node(dual_key)

			for i in range(self.dimension + 1):
				key_diff = numpy.zeros(self.dimension + 1)
				key_diff[i] = 1

				neighbour_dual_key = dual_key + key_diff

				self.dual_graph.create_edge(
					self.dual_graph.node(dual_key),
					self.dual_graph.node(neighbour_dual_key),
					1.0
				)

		for vertex_index in range(self.dimension + 1):
			for dual_key in Simplex.static_simplex_keys(self.dimension, self.size - 1):
				if dual_key[vertex_index] != 0:
					continue
				max_distance_to_center = self.size / (self.dimension + 1)
				middle_simplex_key = max_distance_to_center * (numpy.ones(self.dimension + 1))
				simplex_key = dual_key + 1 / (self.dimension + 1) * numpy.ones(self.dimension + 1)
				# print(dual_key, middle_simplex_key, simplex_key, numpy.absolute(middle_simplex_key - simplex_key), sum(numpy.absolute(middle_simplex_key - simplex_key)), self.size)
				distance_to_center = sum(numpy.absolute(middle_simplex_key - simplex_key)) / 2.
				# base = 2. * (self.size / 3. + 1.)
				base = 2. * (max_distance_to_center + 1)
				offset = numpy.floor(max(distance_to_center - max_distance_to_center, 0))
				# print(max_distance_to_center, dual_key, offset, offset)
				# offset = max(numpy.amax(dual_key) - (2. * self.size / 3.), 0)
				# print(key, numpy.amin(key), numpy.amax(key), max(self.num_hyperplanes - 1 - numpy.amin(key), numpy.amax(key)), offset)
				self.dual_graph.create_edge(
					self.dual_graph.node(dual_key),
					self.dual_graph.node(self.dual_graph.key(vertex_index)),
					base + offset - 1
				)

	def write_dual_graph_instance(self):
		print("write dual graph instance with", self.dual_graph.num_nodes, "nodes and", len(self.dual_graph.edges()), "edges")

		instance = {
			"graph": {
				"directed": False,
				"nodes": [],
				"edges": []
			},
			"nets": [
				{"name": "net_0", "terminals": []}
			]
		}

		for node in self.dual_graph.nodes():
			instance["graph"]["nodes"].append(
				{
					"id": node,
					"key": str(self.dual_graph.key(node))
				}
			)

		for edge in self.dual_graph.edges():
			instance["graph"]["edges"].append(
				{
					"tail": edge["tail"],
					"head": edge["head"],
					"weight": self.dual_graph.edge_weights[edge["id"]]
				}
			)

		for i in range(self.dimension + 1):
			instance["nets"][0]["terminals"].append(i)

		json.dump(instance, open("instance/instance.json", "w"), indent=4)

	def plot_graphs(self, plot_simplex=True, plot_dual=True, plot_dual_terminals=True, plot_ball=True):
		import matplotlib_plotter

		subplot = matplotlib_plotter.init_plot(self.dimension)

		def plot_point(point, colour):
			if self.dimension == 2:
				subplot.plot([point[0]], [point[1]], colour + "o")
			if self.dimension == 3:
				subplot.plot([point[0]], [point[1]], [point[2]], colour + "o")

		if plot_simplex:
			for node in self.simplex_graph.nodes():
				plot_point(Simplex.static_key_to_coordinates(self.simplex_graph.key(node)), "r")

			edge_lines = []

			for edge in self.simplex_graph.edges():
				edge_lines.append(
					[
						Simplex.static_key_to_coordinates(self.simplex_graph.key(edge["tail"])),
						Simplex.static_key_to_coordinates(self.simplex_graph.key(edge["head"]))
					]
				)

			matplotlib_plotter.plot_lines(subplot, edge_lines, color="red", alpha=0.5)

		if plot_dual:
			def is_terminal_node(node):
				return node < self.dimension + 1

			# return node >= self.dual_graph.num_nodes - (self.dimension + 1)

			def compute_dual_coords(node):
				dual_key = self.dual_graph.key(node)
				if is_terminal_node(node):
					return Simplex.static_key_to_coordinates(dual_key)
				else:
					assert sum(dual_key) in [self.size - 1, self.size - 2]
					if sum(dual_key) == self.size - 1:
						return Simplex.static_key_to_coordinates(dual_key) + 1. / (self.dimension + 1) * Simplex.static_key_to_coordinates(numpy.ones(self.dimension + 1))
					if sum(dual_key) == self.size - 2:
						return Simplex.static_key_to_coordinates(dual_key) + 2. / (self.dimension + 1) * Simplex.static_key_to_coordinates(numpy.ones(self.dimension + 1))

			for node in self.dual_graph.nodes():
				if is_terminal_node(node) and not plot_dual_terminals:
					continue

				position = compute_dual_coords(node)
				plot_point(position, "b")

				text_offset = numpy.zeros(self.dimension)
				text_color = "green"
				if self.dimension == 2:
					pass
				if self.dimension == 3:
					text_offset[2] = 0.05

			# matplotlib_plotter.plot_text(subplot, position + text_offset, str(node), color=text_color)

			intern_edge_lines = []
			terminal_edge_lines = []

			for edge in self.dual_graph.edges():
				node_tail = edge["tail"]
				node_head = edge["head"]

				edge_line = [compute_dual_coords(node_tail), compute_dual_coords(node_head)]

				if is_terminal_node(node_tail) or is_terminal_node(node_head):
					if plot_dual_terminals:
						terminal_edge_lines.append(edge_line)
						matplotlib_plotter.plot_text(subplot, 0.5 * (edge_line[0] + edge_line[1]), str(edge["weight"]), color="black")
				else:
					intern_edge_lines.append(edge_line)

			matplotlib_plotter.plot_lines(subplot, terminal_edge_lines, color="green", alpha=0.2)
			matplotlib_plotter.plot_lines(subplot, intern_edge_lines, color="blue")

		if plot_ball:
			ball_lines = []

			for key_diff_1 in self.unit_ball_key_diffs():
				for key_diff_2 in self.unit_ball_key_diffs():
					# print(key_diff_1, key_diff_2, numpy.absolute(key_diff_1 - key_diff_2), sum(numpy.absolute(key_diff_1 - key_diff_2)))
					if sum(numpy.absolute(key_diff_1 - key_diff_2)) != 2:
						continue
					simplex_key_1 = self.size / (self.dimension + 1) * (numpy.ones(self.dimension + 1) + key_diff_1)
					simplex_key_2 = self.size / (self.dimension + 1) * (numpy.ones(self.dimension + 1) + key_diff_2)
					position_1 = Simplex.static_key_to_coordinates(simplex_key_1)
					position_2 = Simplex.static_key_to_coordinates(simplex_key_2)
					# print(key_diff_1, key_diff_2, position_1, position_2)
					ball_lines.append([position_1, position_2])
				# matplotlib_plotter.plot_text(subplot, position_1, str(simplex_key_1), color="black")
				# matplotlib_plotter.plot_text(subplot, position_2, str(simplex_key_2), color="black")

			matplotlib_plotter.plot_lines(subplot, ball_lines, color="green", alpha=1.0, width=3)

		matplotlib_plotter.show_plot()


def main():
	simplex = Simplex(4, 20)

	simplex.write_dual_graph_instance()

	print("dual graph instance is written")

	if simplex.dimension in [0, 1, 2, 3] and simplex.size <= 15:
		simplex.plot_graphs(plot_simplex=True, plot_dual=True, plot_dual_terminals=True, plot_ball=True)
	else:
		print("We do not plot the simplex since the dimension or the size is too large!")


# print(len(simplex.valid_keys), simplex.valid_keys)
# print(len(simplex.unit_ball_key_diffs), simplex.unit_ball_key_diffs)
# print(len(simplex.dual_keys), simplex.dual_keys)

# for vertex_id in range(simplex.num_vertices):
# 	hyperplanes = simplex.compute_hyperplane_keys(numpy.zeros(simplex.num_vertices), vertex_id, 0)
# 	print(vertex_id, len(hyperplanes), hyperplanes)

# simplex.write_dual_graph_instance()

# simplex.plot_graphs(plot_simplex=True, plot_dual=True)


# graph_tool.draw.graph_draw(simplex.dual_graph.graph)


if __name__ == "__main__":
	main()
