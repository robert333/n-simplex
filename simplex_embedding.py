import numpy
import json
import argparse
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

from simplex import Simplex

import matplotlib_plotter


def load_json(path):
	file = open(path)
	result = ""
	for line in file:
		result += line.split("//")[0]
	return json.loads(result)


def compute_simplex_point_2d(point_3d):
	return numpy.array([point_3d[2] / 2 + point_3d[0], point_3d[2] * numpy.sqrt(3) / 2])


def draw_edge_text(pos_1, pos_2, text):
	pyplot.text(0.5 * (pos_1[0] + pos_2[0]), 0.5 * (pos_1[1] + pos_2[1]), text, color="white", ha="center", family="sans-serif", size=12)


def draw_edge_line(ax, pos_1_3d, pos_2_3d, width, color, text):
	pos_1 = compute_simplex_point_2d(pos_1_3d)
	pos_2 = compute_simplex_point_2d(pos_2_3d)

	line = matplotlib.lines.Line2D([pos_1[0], pos_2[0]], [pos_1[1], pos_2[1]], lw=width, alpha=0.3, color=color)
	ax.add_line(line)

	draw_edge_text(pos_1, pos_2, text)


def draw_edge_arrow(ax, pos_1_3d, pos_2_3d, width, color, text):
	pos_1 = compute_simplex_point_2d(pos_1_3d)
	pos_2 = compute_simplex_point_2d(pos_2_3d)

	arrow = matplotlib.patches.Arrow(
		x=pos_1[0],
		y=pos_1[1],
		dx=pos_2[0] - pos_1[0],
		dy=pos_2[1] - pos_1[1],
		width=width,
		alpha=0.3,
		color=color
	)
	ax.add_patch(arrow)

	draw_edge_text(pos_1, pos_2, text)


def parse_embedded_graph(graph_json_path, solution_json_path):
	graph_json = load_json(graph_json_path)
	solution_json = load_json(solution_json_path)

	embedded_graph = {
		"nodes": [],
		"edges": [],
		"num_terminals": len(graph_json["nets"][0]["terminals"])
	}

	for node in graph_json["graph"]["nodes"]:
		node_id = node["id"]
		node_position = numpy.zeros(embedded_graph["num_terminals"])

		for solution_node in solution_json["simplex_embedding"]["nodes"]:
			if solution_node["node"] == str(node_id):
				node_position = numpy.array(solution_node["position"])

		assert len(node_position) == embedded_graph["num_terminals"]

		embedded_graph["nodes"].append(
			{
				"id": node_id,
				"position": node_position
			}
		)

	def parse_integer_value(tail, head):
		for edge in solution_json["optimum_solution"]["edges"]:
			if (edge["tail"] == tail and edge["head"] == head) or (edge["tail"] == head and edge["head"] == tail):
				return 1
		# assert edge["value"] in [0, 1]
		# return edge["value"]
		return 0

	def parse_weight(tail, head):
		for edge in graph_json["graph"]["edges"]:
			if (edge["tail"] == tail and edge["head"] == head) or (edge["tail"] == head and edge["head"] == tail):
				return edge["weight"]
		assert False, [tail, head]

	for edge in solution_json["emcf_edges"]:
		embedded_graph["edges"].append(
			{
				"tail": edge["tail"],
				"head": edge["head"],
				"weight": parse_weight(edge["tail"], edge["head"]),
				"continuous": edge["value"],
				"integer": parse_integer_value(edge["tail"], edge["head"]),
			}
		)

	return embedded_graph


def parse_simplex_size(solution_json_path):
	solution_json = load_json(solution_json_path)

	simplex_size = solution_json["simplex_embedding"]["lambda"]

	if numpy.ceil(simplex_size) != simplex_size:
		print("WARNING: numpy.ceil(simplex_size) != simplex_size =", simplex_size)

	return int(numpy.ceil(simplex_size))


def plot_simplex_embedding(graph_json_path, solution_json_path, output_path):
	embedded_graph = parse_embedded_graph(graph_json_path, solution_json_path)
	simplex_size = parse_simplex_size(solution_json_path)

	simplex_dimension = embedded_graph["num_terminals"] - 1

	assert simplex_dimension in [2, 3]

	subplot = matplotlib_plotter.init_plot(simplex_dimension)

	if simplex_dimension == 3:
		subplot.axes.set_xlim3d(0, simplex_size)
		subplot.axes.set_ylim3d(0, simplex_size)
		subplot.axes.set_zlim3d(0, simplex_size)

	plot_simplex_grid = True
	plot_edge_description = False

	if plot_simplex_grid:
		matplotlib_plotter.plot_lines(subplot, Simplex.compute_simplex_grid_lines(simplex_dimension + 1, simplex_size), alpha=0.1)

	for embedded_node in embedded_graph["nodes"]:
		point_size = None
		text_offset = numpy.zeros(simplex_dimension)
		text_color = None
		if simplex_dimension == 2:
			point_size = 12
			text_color = "white"
		if simplex_dimension == 3:
			point_size = 5
			text_offset[2] = 0.05
			text_color = "green"

		position = Simplex.key_to_3d_coordinates(embedded_node["position"])
		matplotlib_plotter.plot_point(subplot, position, point_size, "blue")
		matplotlib_plotter.plot_text(subplot, position + text_offset, str(embedded_node["id"]), color=text_color)

	used_edge_lines = []
	not_used_edge_lines = []
	flow_arrows = []

	for embedded_edge in embedded_graph["edges"]:
		def get_position(node_id):
			for embedded_node in embedded_graph["nodes"]:
				if embedded_node["id"] == node_id:
					return embedded_node["position"]
			assert False

		position_1 = Simplex.key_to_3d_coordinates(get_position(embedded_edge["tail"]))
		position_2 = Simplex.key_to_3d_coordinates(get_position(embedded_edge["head"]))

		if embedded_edge["integer"] == 1:
			used_edge_lines.append([position_1, position_2])
		else:
			not_used_edge_lines.append([position_1, position_2])

		if round(embedded_edge["continuous"], 9) > 0:
			if plot_edge_description:
				matplotlib_plotter.plot_text(
					subplot,
					0.5 * (position_1 + position_2),
					str(round(embedded_edge["continuous"], 3)) + " / " + str(embedded_edge["weight"])
				)
			flow_arrows.append([position_1, position_2])

	matplotlib_plotter.plot_lines(subplot, used_edge_lines, color="red", width=1)
	matplotlib_plotter.plot_lines(subplot, not_used_edge_lines, color="blue", width=1)
	matplotlib_plotter.plot_arrows(subplot, flow_arrows, color="green", width=1, alpha=0.4)

	if output_path is None:
		matplotlib_plotter.show_plot()
	else:
		pyplot.savefig(output_path + "/figure.png")


def create_argument_parser():
	argument_parser = argparse.ArgumentParser(description="Generate Graph")
	argument_parser.add_argument("instance_path")
	argument_parser.add_argument("solution_path")
	argument_parser.add_argument("--output_path", required=False)
	return argument_parser


def main():
	argument_parser = create_argument_parser()
	arguments = argument_parser.parse_args()

	instance_path = arguments.instance_path
	solution_path = arguments.solution_path
	output_path = arguments.output_path

	plot_simplex_embedding(instance_path, solution_path, output_path)


if __name__ == "__main__":
	main()
