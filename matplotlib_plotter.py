from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Patch3DCollection, Poly3DCollection
import matplotlib.pyplot as pyplot
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Arrow


def init_plot(dimension):
	assert dimension in [2, 3]

	figure = pyplot.figure(figsize=(13, 13))

	if dimension == 2:
		return figure.add_subplot(111)

	if dimension == 3:
		return figure.add_subplot(111, projection="3d")


def show_plot():
	# pyplot.axis("off")
	pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1)
	pyplot.show()


def plot_point(subplot, position, size, color):
	assert len(position) in [2, 3]

	if len(position) == 2:
		subplot.plot([position[0]], [position[1]], color=color, marker='o', markersize=size, alpha=0.6)

	if len(position) == 3:
		subplot.plot([position[0]], [position[1]], [position[2]], color=color, marker='o', markersize=size, alpha=0.6)


def plot_text(subplot, position, text, color="black"):
	assert len(position) in [2, 3]

	if len(position) == 2:
		pyplot.text(position[0], position[1] - 0.05, text, color=color, ha="center", family="sans-serif", size=10)

	if len(position) == 3:
		subplot.text(position[0], position[1], position[2], text, None, color=color, ha="center", family="sans-serif", size=10)


def plot_lines(subplot, lines, color=None, width=None, alpha=None):
	if len(lines) == 0:
		return

	assert len(lines[0]) == 2

	dimension = len(lines[0][0])

	assert dimension in [2, 3]

	if dimension == 2:
		line_collection = LineCollection(
			lines,
			color=color,
			lw=width,
			alpha=alpha
		)
		subplot.add_collection(line_collection)

	if dimension == 3:
		line_3d_collection = Line3DCollection(
			lines,
			color=color,
			lw=width,
			alpha=alpha
		)
		subplot.add_collection3d(line_3d_collection)


def plot_arrows(subplot, arrows, color=None, width=None, alpha=None):
	assert len(arrows) > 0
	assert len(arrows[0]) == 2

	dimension = len(arrows[0][0])

	assert dimension in [2, 3]

	if dimension == 2:
		arrow_patches = []

		for arrow in arrows:
			arrow_patches.append(
				Arrow(
					x=arrow[0][0],
					y=arrow[0][1],
					dx=arrow[1][0] - arrow[0][0],
					dy=arrow[1][1] - arrow[0][1],
					width=width
				)
			)

		subplot.add_collection(
			PatchCollection(
				arrow_patches,
				color=color,
				alpha=alpha
			)
		)

	if dimension == 3:
		for arrow in arrows:
			plot_point(subplot, 0.2 * arrow[0] + 0.8 * arrow[1], size=5, color="green")
