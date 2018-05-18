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

	def exists_node(self, key):
		return tuple(key) in self.key_to_node_map

	def node(self, key):
		assert self.exists_node(key)
		return self.key_to_node_map[tuple(key)]

	def key(self, node):
		return self.node_to_key_map[node]
