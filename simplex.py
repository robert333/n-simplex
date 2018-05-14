import itertools
import numpy
import json
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D


class Graph:
	def __init__(self):
		self.num_nodes = 0
		self.edges_list = []
		self.edge_weights = {}
		self.key_to_node_map = {}
		self.node_to_key_map = {}

	def create_node(self, key):
		node_id = self.num_nodes
		assert tuple(key) not in self.key_to_node_map
		assert node_id not in self.node_to_key_map
		self.key_to_node_map[tuple(key)] = node_id
		self.node_to_key_map[node_id] = key
		self.num_nodes += 1

	def create_edge(self, node_1, node_2, weight):
		assert 0 <= node_1 < self.num_nodes
		assert 0 <= node_2 < self.num_nodes
		assert node_1 != node_2
		edge = {"id": len(self.edges_list), "tail": node_1, "head": node_2, "weight": weight}
		self.edges_list.append(edge)
		self.edge_weights[edge["id"]] = weight

	def nodes(self):
		return range(self.num_nodes)

	def edges(self):
		return self.edges_list

	def node(self, key):
		assert tuple(key) in self.key_to_node_map
		return self.key_to_node_map[tuple(key)]

	def key(self, node):
		return self.node_to_key_map[node]


class Simplex:
	@staticmethod
	def static_assert_if_key_is_not_valid(num_vertices, num_hyperplanes, key):
		assert num_vertices >= 1
		assert num_hyperplanes >= 1
		assert len(key) == num_vertices, [num_vertices, num_hyperplanes, key]
		assert key.dtype == numpy.dtype(int), [key.dtype]
		for i in range(num_vertices):
			assert 0 <= key[i] <= num_hyperplanes, [num_vertices, num_hyperplanes, key]
		assert sum(key) == num_hyperplanes, [num_vertices, num_hyperplanes, key]

	@staticmethod
	def static_compute_next_key(num_vertices, num_hyperplanes, key):
		if key is None:
			first_key = numpy.zeros(num_vertices, dtype=numpy.dtype(int))
			first_key[0] = num_hyperplanes
			return first_key

		Simplex.static_assert_if_key_is_not_valid(num_vertices, num_hyperplanes, key)

		if num_vertices == 1:
			return None

		new_key = key.copy()

		if sum(new_key[1:]) < num_hyperplanes:
			assert new_key[0] > 0
			assert new_key[1] < num_hyperplanes
			new_key[0] -= 1
			new_key[1] += 1
		else:
			new_key[0] += new_key[1]
			new_key[1] -= new_key[1]
			assert new_key[1] == 0
			tmp_key = Simplex.static_compute_next_key(num_vertices - 1, num_hyperplanes, numpy.concatenate([new_key[0:1], new_key[2:]]))
			if tmp_key is None:
				return None
			new_key[0] = tmp_key[0]
			new_key[2:] = tmp_key[1:]

		Simplex.static_assert_if_key_is_not_valid(num_vertices, num_hyperplanes, new_key)
		return new_key

	@staticmethod
	def static_compute_simplex_keys(num_vertices, num_hyperplanes):
		simplex_keys = []
		key = Simplex.static_compute_next_key(num_vertices, num_hyperplanes, None)
		while key is not None:
			simplex_keys.append(key)
			key = Simplex.static_compute_next_key(num_vertices, num_hyperplanes, key)
		return simplex_keys

	@staticmethod
	def static_compute_unit_ball_key_diffs(num_vertices):
		key_diffs = []
		for key in map(list, itertools.product([-1, 0, 1], repeat=num_vertices)):
			if sum(key) == 0 and sum(abs(x) for x in key) == 2:
				key_diffs.append(numpy.array(key, dtype=numpy.dtype(int)))
		return key_diffs

	@staticmethod
	def static_unit_vectors_3d():
		return [
			numpy.array([0., 0., 0.]),
			numpy.array([1., 0., 0.]),
			numpy.array([1. / 2., numpy.sqrt(3.) / 2., 0.]),
			numpy.array([1. / 2., 1. / (2. * numpy.sqrt(3.)), numpy.sqrt(2.) / numpy.sqrt(3.)])
		]

	@staticmethod
	def key_to_3d_coordinates(key):
		assert len(key) <= 4
		unit_vectors = Simplex.static_unit_vectors_3d()
		coordinates = numpy.zeros(3)
		for i in range(len(key)):
			coordinates += key[i] * unit_vectors[i]
		return coordinates[:len(key) - 1]

	@staticmethod
	def compute_simplex_grid_lines(num_vertices, num_hyperplanes):
		assert num_vertices in [3, 4]

		simplex_keys = Simplex.static_compute_simplex_keys(num_vertices, num_hyperplanes)

		positive_unit_ball_key_diffs = []

		for unit_ball_key_diff in Simplex.static_compute_unit_ball_key_diffs(num_vertices):
			if numpy.where(unit_ball_key_diff == 1)[0] < numpy.where(unit_ball_key_diff == -1)[0]:
				positive_unit_ball_key_diffs.append(unit_ball_key_diff)

		grid_lines = []

		for simplex_key in simplex_keys:
			for positive_unit_ball_key_diff in positive_unit_ball_key_diffs:
				if all(0 <= x <= num_hyperplanes for x in (simplex_key - positive_unit_ball_key_diff)):
					continue

				if not all(0 <= x <= num_hyperplanes for x in (simplex_key + positive_unit_ball_key_diff)):
					continue

				i = numpy.where(positive_unit_ball_key_diff == 1)[0][0]
				j = numpy.where(positive_unit_ball_key_diff == -1)[0][0]

				if num_hyperplanes - simplex_key[i] <= simplex_key[j] - 0:
					other_key = simplex_key + (num_hyperplanes - simplex_key[i]) * positive_unit_ball_key_diff
				else:
					assert num_hyperplanes - simplex_key[i] > simplex_key[j] - 0
					other_key = simplex_key + (simplex_key[j] - 0) * positive_unit_ball_key_diff

				# print(simplex_key, i, j, other_key, positive_unit_ball_key_diff)

				assert not numpy.array_equal(simplex_key, other_key)

				# print(simplex_key, other_key)

				grid_lines.append([Simplex.key_to_3d_coordinates(simplex_key), Simplex.key_to_3d_coordinates(other_key)])

		return grid_lines

	def __init__(self, dimension, num_hyperplanes):
		self.dimension = dimension
		self.num_vertices = dimension + 1
		self.num_hyperplanes = num_hyperplanes

		self.valid_keys = Simplex.static_compute_simplex_keys(self.num_vertices, self.num_hyperplanes)
		self.unit_ball_key_diffs = Simplex.static_compute_unit_ball_key_diffs(self.num_vertices)

		self.dual_keys = self.compute_dual_keys()

		self.simplex_graph = Graph()
		self.dual_graph = Graph()

		self.create_simplex_graph()
		self.create_dual_graph()

	def assert_if_key_is_not_valid(self, key):
		Simplex.static_assert_if_key_is_not_valid(self.num_vertices, self.num_hyperplanes, key)

	def compute_next_key(self, key):
		return Simplex.static_compute_next_key(self.num_vertices, self.num_hyperplanes, key)

	def is_simplex_key_in_the_room(self, dual_key, simplex_key):
		for i in range(self.num_vertices):
			if not (dual_key[i] <= simplex_key[i] <= dual_key[i] + 1):
				return False
		return True

	def compute_hyperplane_keys(self, dual_key, hyperplane_dimension=None, dimension_orientation=None):
		if hyperplane_dimension is not None:
			assert dimension_orientation is not None
			return [
				simplex_key for simplex_key in self.valid_keys
				if (
						simplex_key[hyperplane_dimension] == dual_key[hyperplane_dimension] + dimension_orientation
						and self.is_simplex_key_in_the_room(dual_key, simplex_key)
				)
			]

		return [simplex_key for simplex_key in self.valid_keys if self.is_simplex_key_in_the_room(dual_key, simplex_key)]

	def has_dual_neighbour_room(self, dual_key, hyperplane_dimension, dimension_orientation):
		return len(self.compute_hyperplane_keys(dual_key, hyperplane_dimension, dimension_orientation)) >= self.num_vertices - 1

	def dual_vertex_to_coordinates(self, dual_vertex):
		hyperplane_keys = self.compute_hyperplane_keys(self.dual_graph.key(dual_vertex))
		coordinates_list = [Simplex.key_to_3d_coordinates(simplex_key) for simplex_key in hyperplane_keys]
		assert len(coordinates_list) > 0, self.dual_graph.key(dual_vertex)
		coordinates = numpy.zeros(len(coordinates_list[0]))
		for tmp in coordinates_list:
			coordinates += tmp
		coordinates /= len(coordinates_list)
		return coordinates

	def compute_dual_keys(self):
		def is_dual_key_valid(dual_key):
			for i in range(self.num_vertices):
				if not (0 <= dual_key[i] <= self.num_hyperplanes - 1):
					return False
			return True

		first_dual_key = numpy.zeros(self.num_vertices, dtype=numpy.dtype(int))
		first_dual_key[0] = self.num_hyperplanes - 1

		dual_keys = [first_dual_key]
		room_stack = [first_dual_key]

		while room_stack:
			dual_key = room_stack.pop()
			for i in range(self.num_vertices):
				for k in [0, 1]:
					if self.has_dual_neighbour_room(dual_key, i, k):
						neighbour_key = dual_key.copy()
						if k == 0:
							neighbour_key[i] += -1
						else:
							assert k == 1
							neighbour_key[i] += +1
						assert sum(neighbour_key) < self.num_hyperplanes
						if is_dual_key_valid(neighbour_key) and not any((neighbour_key == x).all() for x in dual_keys) and sum(neighbour_key) in [self.num_hyperplanes - 1, self.num_hyperplanes - 2]:
							dual_keys.append(neighbour_key)
							room_stack.append(neighbour_key)

		return dual_keys

	def create_simplex_graph(self):
		for key in self.valid_keys:
			self.simplex_graph.create_node(key)

		for key in self.valid_keys:
			for key_diff in self.unit_ball_key_diffs:
				if numpy.where(key_diff == 1)[0] < numpy.where(key_diff == -1)[0]:
					neighbour_key = key + key_diff
					if all(x >= 0 for x in neighbour_key):
						node_1 = self.simplex_graph.node(key)
						node_2 = self.simplex_graph.node(key + key_diff)
						self.simplex_graph.create_edge(node_1, node_2, 1.0)

	def create_dual_graph(self):
		terminals = {}

		for vertex_id in range(self.num_vertices):
			key = numpy.zeros(self.num_vertices, dtype=numpy.dtype(float))
			for i in range(self.num_vertices):
				if i == vertex_id:
					key[i] = -1 * self.num_hyperplanes
				else:
					key[i] = 2 * self.num_hyperplanes / (self.num_vertices - 1)
			terminals[vertex_id] = key
			self.dual_graph.create_node(key)

		for key in self.dual_keys:
			self.dual_graph.create_node(key)

		def lexicographic_comparison(a, b):
			assert len(a) == len(b)
			for i in range(len(a)):
				if a[i] != b[i]:
					return a[i] < b[i]

		for key_1 in self.dual_keys:
			for key_2 in self.dual_keys:
				if not lexicographic_comparison(key_1, key_2):
					continue
				if numpy.sum(numpy.absolute(key_1 - key_2)) == 1:
					node_1 = self.dual_graph.node(key_1)
					node_2 = self.dual_graph.node(key_2)
					self.dual_graph.create_edge(node_1, node_2, 1.0)

		for key in self.dual_keys:
			for vertex_id in range(self.num_vertices):
				if key[vertex_id] == 0 and self.has_dual_neighbour_room(key, vertex_id, 0) and sum(key) == self.num_hyperplanes - 1:
					node = self.dual_graph.node(key)
					terminal = self.dual_graph.node(terminals[vertex_id])
					offset = max(numpy.amax(key) - (2. * self.num_hyperplanes / 3.), 0)
					# print(key, numpy.amin(key), numpy.amax(key), max(self.num_hyperplanes - 1 - numpy.amin(key), numpy.amax(key)), offset)
					self.dual_graph.create_edge(node, terminal, 2. * (self.num_hyperplanes / 3. + 1.) + offset)  # 2 * (self.num_hyperplanes - 1))

	def write_dual_graph_instance(self):
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

		for i in range(self.num_vertices):
			instance["nets"][0]["terminals"].append(i)

		json.dump(instance, open("instance/instance.json", "w"), indent=4)

	def plot_graphs(self, plot_simplex=True, plot_dual=True):
		import matplotlib_plotter

		subplot = matplotlib_plotter.init_plot(self.dimension)

		def plot_point(point, colour):
			if self.dimension == 2:
				subplot.plot([point[0]], [point[1]], colour + "o")
			if self.dimension == 3:
				subplot.plot([point[0]], [point[1]], [point[2]], colour + "o")

		if plot_simplex:
			for node in self.simplex_graph.nodes():
				plot_point(Simplex.key_to_3d_coordinates(self.simplex_graph.key(node)), "r")

			edge_lines = []

			for edge in self.simplex_graph.edges():
				edge_lines.append(
					[
						Simplex.key_to_3d_coordinates(self.simplex_graph.key(edge["tail"])),
						Simplex.key_to_3d_coordinates(self.simplex_graph.key(edge["head"]))
					]
				)

			matplotlib_plotter.plot_lines(subplot, edge_lines, color="red", alpha=0.5)

		if plot_dual:
			def is_terminal_node(node):
				return node < self.num_vertices

			def compute_dual_coords(node):
				if is_terminal_node(node):
					return Simplex.key_to_3d_coordinates(self.dual_graph.key(node))
				else:
					return self.dual_vertex_to_coordinates(node)

			for node in self.dual_graph.nodes():
				position = compute_dual_coords(node)
				plot_point(position, "b")

				text_offset = numpy.zeros(self.dimension)
				text_color = "green"
				if self.dimension == 2:
					pass
				if self.dimension == 3:
					text_offset[2] = 0.05

				matplotlib_plotter.plot_text(subplot, position + text_offset, str(node), color=text_color)

			intern_edge_lines = []
			terminal_edge_lines = []

			for edge in self.dual_graph.edges():
				node_tail = edge["tail"]
				node_head = edge["head"]

				edge_line = [compute_dual_coords(node_tail), compute_dual_coords(node_head)]

				if is_terminal_node(node_tail) or is_terminal_node(node_head):
					terminal_edge_lines.append(edge_line)
					matplotlib_plotter.plot_text(subplot, 0.5 * (edge_line[0] + edge_line[1]), str(edge["weight"]), color="black")
				else:
					intern_edge_lines.append(edge_line)

			matplotlib_plotter.plot_lines(subplot, terminal_edge_lines, color="green", alpha=0.2)
			matplotlib_plotter.plot_lines(subplot, intern_edge_lines, color="blue")

		matplotlib_plotter.show_plot()


def main():
	simplex = Simplex(3, 15)

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
