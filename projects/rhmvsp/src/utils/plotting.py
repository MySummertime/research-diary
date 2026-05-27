# -*- coding: utf-8 -*-
"""
[View Layer] High-Quality Plotting & Visualization Utilities for RHMVSP.
Adheres strictly to top-tier academic journal standards (600 DPI, L-shaped axes, Times New Roman, dynamic arc avoidance).
"""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from ..core.network import MultimodalNetwork
from config import ColorPalette, PlotProperties, get_color_by_key


class Visualizer:
	"""Handles publication-quality visualization of multimodal networks and scheduled routes."""

	def __init__(self, config):
		self.config = config
		self._apply_academic_style()

		self.default_colors = ColorPalette.DEFAULT_COLOR
		self.network_colors = ColorPalette.NETWORK_TOPOLOGY
		self.task_colors = ColorPalette.TASK_LOOP

		self.styles = {
			"figure_size": PlotProperties.FIG_SIZE,
			"node": {
				"hub": {
					"color": get_color_by_key(self.network_colors, "HUB"),
					"shape": PlotProperties.NODE_SHAPE,
					"size": PlotProperties.NODE_SIZE_HUB,
					"edgecolor": PlotProperties.MARKER_EDGE_COLOR,
					"linewidth": PlotProperties.NODE_LINE_WIDTH,
				},
				"non_hub": {
					"color": get_color_by_key(self.network_colors, "NON_HUB"),
					"shape": PlotProperties.NODE_SHAPE,
					"size": PlotProperties.NODE_SIZE_SPOKE,
					"edgecolor": PlotProperties.MARKER_EDGE_COLOR,
					"linewidth": PlotProperties.NODE_LINE_WIDTH,
				},
			},
			"edge": {
				"road": {
					"color": get_color_by_key(self.network_colors, "ROAD"),
					"width": PlotProperties.LINE_WIDTH_ROAD,
					"style": PlotProperties.STYLE_ROAD,
					"alpha": PlotProperties.ALPHA_ROAD,
				},
				"railway": {
					"color": get_color_by_key(self.network_colors, "RAILWAY"),
					"width": PlotProperties.LINE_WIDTH_RAIL,
					"style": PlotProperties.STYLE_RAIL,
					"alpha": PlotProperties.ALPHA_RAIL,
				},
			},
			"font": {
				"size": PlotProperties.FONT_SIZE_NODE_ID,
				"color": PlotProperties.FONT_COLOR_NODE_ID,
				"weight": PlotProperties.FONT_WEIGHT_BOLD,
			},
		}

	def _apply_academic_style(self):
		"""Apply global Matplotlib settings for publication standards."""
		plt.rcParams.update(
			{
				"font.family": PlotProperties.FONT_FAMILY,
				"font.serif": [PlotProperties.FONT_FAMILY, "Times New Roman", "DejaVu Serif"],
				"mathtext.fontset": "cm",
				"font.size": PlotProperties.FONT_SIZE_BASE,
				"axes.labelsize": PlotProperties.FONT_SIZE_AXIS_LABEL,
				"axes.titlesize": PlotProperties.FONT_SIZE_TITLE,
				"xtick.labelsize": PlotProperties.FONT_SIZE_TICK,
				"ytick.labelsize": PlotProperties.FONT_SIZE_TICK,
				"legend.fontsize": PlotProperties.FONT_SIZE_LEGEND,
				"xtick.direction": "in",
				"ytick.direction": "in",
				"lines.linewidth": 2,
				"axes.grid": True,
				"grid.linestyle": PlotProperties.GRID_LINE_STYLE,
				"grid.alpha": PlotProperties.GRID_ALPHA,
				"savefig.dpi": PlotProperties.DPI,
				"savefig.bbox": "tight",
			}
		)

	def _format_axes(self, ax):
		"""Format axes with L-shaped minimalist spine and inward ticks."""
		ax.grid(True, linestyle=PlotProperties.GRID_LINE_STYLE, alpha=PlotProperties.GRID_ALPHA)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["bottom"].set_linewidth(PlotProperties.SPINE_LINE_WIDTH)
		ax.spines["left"].set_linewidth(PlotProperties.SPINE_LINE_WIDTH)
		ax.tick_params(
			direction="in", length=PlotProperties.TICK_LENGTH, width=PlotProperties.TICK_WIDTH
		)
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_fontname(PlotProperties.FONT_FAMILY)

	def _calculate_edge_styles(self, edgelist: List[Tuple[int, int]]) -> List[str]:
		"""Calculate connection styles with incremental curvature to ensure arcs never overlap."""
		styles = [""] * len(edgelist)
		pair_counts = {}
		for i, (u, v) in enumerate(edgelist):
			pair = frozenset({u, v})
			rank = pair_counts.get(pair, 0)
			pair_counts[pair] = rank + 1
			rad = PlotProperties.CURVATURE_STEP_RAD * rank + PlotProperties.CURVATURE_BASE_RAD
			styles[i] = f"arc3,rad={rad:.2f}"
		return styles

	def plot_network(self, network: MultimodalNetwork, output_path: str):
		"""Plot the complete physical multimodal network topology (unfiltered)."""
		fig, ax = plt.subplots(figsize=self.styles["figure_size"])
		pos = {}
		for n, data in network.graph.nodes(data=True):
			pos[n] = (data.get("x", np.random.uniform(0, 100)), data.get("y", np.random.uniform(0, 100)))

		road_edges = set()
		rail_edges = set()
		for arc in network.arcs.values():
			if arc.is_road:
				road_edges.add(frozenset({arc.i, arc.j}))
			elif arc.is_rail:
				rail_edges.add(frozenset({arc.i, arc.j}))

		road_edgelist = [tuple(pair) for pair in road_edges]
		rail_edgelist = [tuple(pair) for pair in rail_edges]

		if road_edgelist:
			road_styles = self._calculate_edge_styles(road_edgelist)
			style_r = self.styles["edge"]["road"]
			for i, (u, v) in enumerate(road_edgelist):
				patches = nx.draw_networkx_edges(network.graph, pos, ax=ax, edgelist=[(u, v)], edge_color=style_r["color"], width=style_r["width"], style=style_r["style"], alpha=style_r["alpha"], arrows=True, arrowstyle="-", connectionstyle=road_styles[i])
				if patches:
					for p in patches:
						p.set_linestyle(style_r["style"])

		if rail_edgelist:
			rail_styles = self._calculate_edge_styles(rail_edgelist)
			style_m = self.styles["edge"]["railway"]
			for i, (u, v) in enumerate(rail_edgelist):
				patches = nx.draw_networkx_edges(network.graph, pos, ax=ax, edgelist=[(u, v)], edge_color=style_m["color"], width=style_m["width"], style=style_m["style"], alpha=style_m["alpha"], arrows=True, arrowstyle="-", connectionstyle=rail_styles[i])
				if patches:
					for p in patches:
						p.set_capstyle("butt")
						p.set_linestyle((0, (8, 8)))

		transfer_node_ids = set(network.transfer_nodes.keys())
		all_node_ids = set(network.graph.nodes())
		hub_nodes = list(transfer_node_ids)
		non_hub_nodes = list(all_node_ids - transfer_node_ids)

		style_hub = self.styles["node"]["hub"]
		if hub_nodes:
			nx.draw_networkx_nodes(network.graph, pos, ax=ax, nodelist=hub_nodes, node_color=style_hub["color"], node_shape=style_hub["shape"], node_size=style_hub["size"], edgecolors=style_hub["edgecolor"], linewidths=style_hub["linewidth"])

		style_non = self.styles["node"]["non_hub"]
		if non_hub_nodes:
			nx.draw_networkx_nodes(network.graph, pos, ax=ax, nodelist=non_hub_nodes, node_color=style_non["color"], node_shape=style_non["shape"], node_size=style_non["size"], edgecolors=style_non["edgecolor"], linewidths=style_non["linewidth"])

		id_labels = {n: str(n) for n in network.graph.nodes()}
		nx.draw_networkx_labels(network.graph, pos, labels=id_labels, ax=ax, font_family=PlotProperties.FONT_FAMILY, font_size=self.styles["font"]["size"], font_color=PlotProperties.FONT_COLOR_NODE_ID, font_weight=PlotProperties.FONT_WEIGHT_BOLD)

		self._format_axes(ax)
		ax.set_xlabel(PlotProperties.LABEL_X_AXIS, fontsize=PlotProperties.FONT_SIZE_AXIS_LABEL, fontweight=PlotProperties.FONT_WEIGHT_BOLD, fontfamily=PlotProperties.FONT_FAMILY)
		ax.set_ylabel(PlotProperties.LABEL_Y_AXIS, fontsize=PlotProperties.FONT_SIZE_AXIS_LABEL, fontweight=PlotProperties.FONT_WEIGHT_BOLD, fontfamily=PlotProperties.FONT_FAMILY)

		legend_handles = [
			Line2D([0], [0], marker=PlotProperties.NODE_SHAPE, color=PlotProperties.LEGEND_MARKER_FACE_WHITE, label=PlotProperties.LABEL_LEGEND_HUB, markerfacecolor=style_hub["color"], markeredgecolor=style_hub["edgecolor"], markersize=PlotProperties.LEGEND_MARKER_SIZE_HUB, markeredgewidth=PlotProperties.LEGEND_MARKER_EDGE_WIDTH_HUB),
			Line2D([0], [0], marker=PlotProperties.NODE_SHAPE, color=PlotProperties.LEGEND_MARKER_FACE_WHITE, label=PlotProperties.LABEL_LEGEND_SPOKE, markerfacecolor=style_non["color"], markeredgecolor=style_non["edgecolor"], markersize=PlotProperties.LEGEND_MARKER_SIZE_SPOKE, markeredgewidth=PlotProperties.LEGEND_MARKER_EDGE_WIDTH_SPOKE),
			Line2D([0], [0], color=get_color_by_key(self.network_colors, "RAILWAY"), lw=PlotProperties.LINE_WIDTH_RAIL, linestyle=PlotProperties.STYLE_RAIL, label=PlotProperties.LABEL_LEGEND_RAIL),
			Line2D([0], [0], color=get_color_by_key(self.network_colors, "ROAD"), lw=PlotProperties.LINE_WIDTH_ROAD, linestyle=PlotProperties.STYLE_ROAD, label=PlotProperties.LABEL_LEGEND_ROAD),
		]
		ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, borderaxespad=0, frameon=False, prop={"family": PlotProperties.FONT_FAMILY, "size": PlotProperties.FONT_SIZE_LEGEND})

		base_dir, base_name = os.path.split(output_path)
		name_root, _ = os.path.splitext(base_name)
		png_path = os.path.join(base_dir, f"{name_root}.png")
		tiff_path = os.path.join(base_dir, f"{name_root}.tiff")
		plt.tight_layout()
		plt.savefig(png_path, dpi=PlotProperties.DPI, transparent=True, bbox_inches="tight", pad_inches=PlotProperties.PAD_INCHES)
		plt.savefig(tiff_path, format="tiff", dpi=PlotProperties.DPI, bbox_inches="tight", pad_inches=PlotProperties.PAD_INCHES)
		plt.close()

	def plot_solution(self, network: MultimodalNetwork, solution: dict, output_path: str):
		"""Plot the multimodal transport network and optimized vehicle routes."""
		if not solution or ("vehicle_schedule" not in solution and "transport_routes" not in solution):
			return

		fig, ax = plt.subplots(figsize=self.styles["figure_size"])

		# 1. Extract raw physical coordinates from network graph nodes
		pos = {}
		for n, data in network.graph.nodes(data=True):
			pos[n] = (
				data.get("x", np.random.uniform(0, 100)),
				data.get("y", np.random.uniform(0, 100)),
			)

		# 2. Draw background multimodal topology (only arcs used by scheduled routes)
		v_schedules = solution.get("transport_routes", solution.get("vehicle_schedule", []))
		used_edges = set()
		for v_task in v_schedules:
			for arc_item in v_task.get("arcs", []):
				if isinstance(arc_item, dict):
					used_edges.add(frozenset({arc_item['i'], arc_item['j']}))
				else:
					used_edges.add(frozenset({arc_item[0], arc_item[1]}))

		bg_style_map = {}
		road_edges_all = set()
		rail_edges_all = set()
		for arc in network.arcs.values():
			pair = frozenset({arc.i, arc.j})
			if arc.is_road:
				road_edges_all.add(pair)
			elif arc.is_rail:
				rail_edges_all.add(pair)

		road_edgelist = [tuple(pair) for pair in road_edges_all]
		rail_edgelist = [tuple(pair) for pair in rail_edges_all]

		# Draw road infrastructure
		if road_edgelist:
			road_styles = self._calculate_edge_styles(road_edgelist)
			style_r = self.styles["edge"]["road"]
			for i, (u, v) in enumerate(road_edgelist):
				bg_style_map[(u, v, 1)] = road_styles[i]
				bg_style_map[(v, u, 1)] = road_styles[i]
				pair = frozenset({u, v})
				if pair in used_edges:
					continue  # Skip drawing topology background arc; it will be fully represented by active colored route!
				patches = nx.draw_networkx_edges(
					network.graph,
					pos,
					ax=ax,
					edgelist=[(u, v)],
					edge_color=style_r["color"],
					width=style_r["width"],
					style=style_r["style"],
					alpha=0.10,
					connectionstyle=road_styles[i],
					arrows=True,
					arrowstyle="-",
				)
				if patches:
					for p in patches:
						p.set_linestyle(style_r["style"])

		# Draw rail infrastructure
		if rail_edgelist:
			rail_styles = self._calculate_edge_styles(rail_edgelist)
			style_m = self.styles["edge"]["railway"]
			for i, (u, v) in enumerate(rail_edgelist):
				bg_style_map[(u, v, 2)] = rail_styles[i]
				bg_style_map[(v, u, 2)] = rail_styles[i]
				pair = frozenset({u, v})
				if pair in used_edges:
					continue  # Skip drawing topology background arc
				patches = nx.draw_networkx_edges(
					network.graph,
					pos,
					ax=ax,
					edgelist=[(u, v)],
					edge_color=style_m["color"],
					width=style_m["width"],
					style=style_m["style"],
					alpha=0.10,
					connectionstyle=rail_styles[i],
					arrows=True,
					arrowstyle="-",
				)
				if patches:
					for p in patches:
						p.set_capstyle("butt")
						p.set_linestyle(PlotProperties.DASH_PATTERN_RAIL)

		# 3. Draw active scheduled vehicle routes
		# Sort OD pairs numerically (OD_1, OD_2, ...)
		unique_od_names = sorted(list(set(t["od_pair"] for t in v_schedules)), key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
		od_color_map = {name: self.task_colors[i % len(self.task_colors)] for i, name in enumerate(unique_od_names)}
		
		legend_handles = []
		for od_name in unique_od_names:
			color = od_color_map[od_name]
			task_num = od_name.split("_")[1] if "_" in od_name else od_name
			legend_handles.append(Line2D([0], [0], color=color, lw=3.0, label=f"{PlotProperties.LABEL_LEGEND_TASK_PREFIX}{task_num}"))

		route_summary = {}  # (u, v, mode, od_name) -> list of path indices
		unique_paths = {}   # (od_name, tuple_of_arcs) -> path_idx

		for v_task in v_schedules:
			od_name = v_task["od_pair"]
			route_arcs = v_task.get("arcs", [])
			# Extract just the (u, v, mode) for the path identity
			path_identity = tuple((arc['i'], arc['j'], arc['mode']) if isinstance(arc, dict) else (arc[0], arc[1], arc[2]) for arc in route_arcs)
			path_key = (od_name, path_identity)
			
			if path_key not in unique_paths:
				unique_paths[path_key] = len(unique_paths)
			path_idx = unique_paths[path_key]

			for arc_item in route_arcs:
				if isinstance(arc_item, dict):
					u, v, mode = arc_item['i'], arc_item['j'], arc_item['mode']
				else:
					u, v, mode = arc_item[:3]
				key = (u, v, mode, od_name)
				if key not in route_summary:
					route_summary[key] = []
				route_summary[key].append(path_idx)

		if route_summary:
			unique_keys = list(route_summary.keys())
			pair_route_counter = {}

			for key in unique_keys:
				u, v, mode, od_name = key
				pair = frozenset({u, v})
				rank = pair_route_counter.get(pair, 0)
				pair_route_counter[pair] = rank + 1

				veh_ids = sorted(list(set(route_summary[key])))
				color = od_color_map[od_name]
				base_conn = bg_style_map.get((u, v, mode), "arc3,rad=0.10")

				base_rad = 0.10
				if "rad=" in base_conn:
					try:
						base_rad = float(base_conn.split("rad=")[1])
					except ValueError:
						pass
				
				# Dynamic rank-based expansion so parallel routes form distinct rainbow curves
				adjusted_rad = base_rad + (rank - 1) * 0.15
				adjusted_conn_style = f"arc3,rad={adjusted_rad:.2f}"

				if mode == 1:
					# Standard solid road route
					nx.draw_networkx_edges(
						network.graph, pos, ax=ax, edgelist=[(u, v)], edge_color=color,
						width=PlotProperties.LINE_WIDTH_ROUTE, style="-",
						alpha=PlotProperties.ALPHA_ROUTE, arrows=True,
						arrowstyle=PlotProperties.ARROW_STYLE, arrowsize=PlotProperties.ARROW_SIZE,
						connectionstyle=adjusted_conn_style,
					)
				else:
					# Mode 2 Railway route: Two-pass rendering to guarantee flawless visible dashed lines!
					# Pass 1: Draw the curved line body without arrows so dash pattern is perfectly respected by Matplotlib
					patches = nx.draw_networkx_edges(
						network.graph, pos, ax=ax, edgelist=[(u, v)], edge_color=color,
						width=PlotProperties.LINE_WIDTH_ROUTE, style="--",
						alpha=PlotProperties.ALPHA_ROUTE, arrows=True,
						arrowstyle="-", connectionstyle=adjusted_conn_style,
					)
					if patches:
						for p in patches:
							p.set_capstyle("butt")
							p.set_linestyle(PlotProperties.DASH_PATTERN_RAIL)
					
					# Pass 2: Draw ONLY the solid arrow tip at the end by completely eliminating the line body edges
					patches2 = nx.draw_networkx_edges(
						network.graph, pos, ax=ax, edgelist=[(u, v)], edge_color=color,
						width=0.0, style="-",
						alpha=PlotProperties.ALPHA_ROUTE, arrows=True,
						arrowstyle=PlotProperties.ARROW_STYLE, arrowsize=PlotProperties.ARROW_SIZE,
						connectionstyle=adjusted_conn_style,
					)
					if patches2:
						for p2 in patches2:
							p2.set_linewidth(0)
							p2.set_edgecolor("none")

				# Format vehicle index label
				if len(veh_ids) > 2 and veh_ids[-1] - veh_ids[0] == len(veh_ids) - 1:
					veh_label = f"{PlotProperties.LABEL_VEH_PREFIX}{veh_ids[0]}-{veh_ids[-1]}"
				elif len(veh_ids) > 4:
					veh_label = f"{PlotProperties.LABEL_VEH_PREFIX}{veh_ids[0]}-{veh_ids[-1]} ({len(veh_ids)}v)"
				else:
					veh_label = PlotProperties.LABEL_VEH_PREFIX + ",".join(str(x) for x in veh_ids)

				# Calculate text annotation center with guaranteed absolute normal separation
				x_u, y_u = pos[u]
				x_v, y_v = pos[v]
				dx = x_v - x_u
				dy = y_v - y_u

				edge_len = np.hypot(dx, dy)
				if edge_len < 1e-3:
					edge_len = 1.0
				nx_norm = (-dy) / edge_len
				ny_norm = dx / edge_len

				# Push each vehicle tag outwards with absolute distance (3.8 units per rank)
				abs_offset = 2.5 + (rank - 1) * 3.8
				mx = (x_u + x_v) / 2.0 + abs_offset * nx_norm
				my = (y_u + y_v) / 2.0 + abs_offset * ny_norm

				ax.text(
					mx,
					my,
					veh_label,
					color=color,
					fontfamily=PlotProperties.FONT_FAMILY,
					fontweight=PlotProperties.FONT_WEIGHT_BOLD,
					fontsize=PlotProperties.FONT_SIZE_VEH_LABEL,
					ha="center",
					va="center",
					zorder=6,
					bbox=PlotProperties.TRANSPARENT_BBOX,
				)

		# 4. Draw Hub (Transfer Nodes) and Non-Hub Nodes
		transfer_node_ids = set(network.transfer_nodes.keys())
		all_node_ids = set(network.graph.nodes())
		hub_nodes = list(transfer_node_ids)
		non_hub_nodes = list(all_node_ids - transfer_node_ids)

		style_hub = self.styles["node"]["hub"]
		if hub_nodes:
			nx.draw_networkx_nodes(
				network.graph,
				pos,
				ax=ax,
				nodelist=hub_nodes,
				node_color=style_hub["color"],
				node_shape=style_hub["shape"],
				node_size=style_hub["size"],
				edgecolors=style_hub["edgecolor"],
				linewidths=style_hub["linewidth"],
			)

		style_non = self.styles["node"]["non_hub"]
		if non_hub_nodes:
			nx.draw_networkx_nodes(
				network.graph,
				pos,
				ax=ax,
				nodelist=non_hub_nodes,
				node_color=style_non["color"],
				node_shape=style_non["shape"],
				node_size=style_non["size"],
				edgecolors=style_non["edgecolor"],
				linewidths=style_non["linewidth"],
			)

		# 5. Draw node labels & ID numbers
		# ID numbers in the center
		id_labels = {n: str(n) for n in network.graph.nodes()}
		nx.draw_networkx_labels(
			network.graph,
			pos,
			labels=id_labels,
			ax=ax,
			font_family=PlotProperties.FONT_FAMILY,
			font_size=self.styles["font"]["size"],
			font_color=PlotProperties.FONT_COLOR_NODE_ID,
			font_weight=PlotProperties.FONT_WEIGHT_BOLD,
		)

		# 6. Formatting and Legend
		self._format_axes(ax)
		ax.set_xlabel(
			PlotProperties.LABEL_X_AXIS,
			fontsize=PlotProperties.FONT_SIZE_AXIS_LABEL,
			fontweight=PlotProperties.FONT_WEIGHT_BOLD,
			fontfamily=PlotProperties.FONT_FAMILY,
		)
		ax.set_ylabel(
			PlotProperties.LABEL_Y_AXIS,
			fontsize=PlotProperties.FONT_SIZE_AXIS_LABEL,
			fontweight=PlotProperties.FONT_WEIGHT_BOLD,
			fontfamily=PlotProperties.FONT_FAMILY,
		)

		# Add network infrastructure to legend
		legend_handles.insert(
			0,
			Line2D(
				[0],
				[0],
				marker=PlotProperties.NODE_SHAPE,
				color=PlotProperties.LEGEND_MARKER_FACE_WHITE,
				label=PlotProperties.LABEL_LEGEND_HUB,
				markerfacecolor=style_hub["color"],
				markeredgecolor=style_hub["edgecolor"],
				markersize=PlotProperties.LEGEND_MARKER_SIZE_HUB,
				markeredgewidth=PlotProperties.LEGEND_MARKER_EDGE_WIDTH_HUB,
			),
		)
		legend_handles.insert(
			1,
			Line2D(
				[0],
				[0],
				marker=PlotProperties.NODE_SHAPE,
				color=PlotProperties.LEGEND_MARKER_FACE_WHITE,
				label=PlotProperties.LABEL_LEGEND_SPOKE,
				markerfacecolor=style_non["color"],
				markeredgecolor=style_non["edgecolor"],
				markersize=PlotProperties.LEGEND_MARKER_SIZE_SPOKE,
				markeredgewidth=PlotProperties.LEGEND_MARKER_EDGE_WIDTH_SPOKE,
			),
		)
		legend_handles.insert(
			2,
			Line2D(
				[0],
				[0],
				color=get_color_by_key(self.network_colors, "RAILWAY"),
				lw=PlotProperties.LINE_WIDTH_RAIL,
				linestyle=PlotProperties.DASH_PATTERN_RAIL,
				label=PlotProperties.LABEL_LEGEND_RAIL,
			),
		)
		legend_handles.insert(
			3,
			Line2D(
				[0],
				[0],
				color=get_color_by_key(self.network_colors, "ROAD"),
				lw=PlotProperties.LINE_WIDTH_ROAD,
				linestyle=PlotProperties.STYLE_ROAD,
				label=PlotProperties.LABEL_LEGEND_ROAD,
			),
		)

		dynamic_ncol = max(4, (len(legend_handles) + 1) // 2)
		ax.legend(
			handles=legend_handles,
			loc="lower left",
			bbox_to_anchor=(0.0, 1.02),
			ncol=dynamic_ncol,
			borderaxespad=0,
			frameon=False,
			shadow=False,
			prop={"family": PlotProperties.FONT_FAMILY, "size": PlotProperties.FONT_SIZE_LEGEND},
		)

		# 7. Save outputs in multiple publication formats (PNG & TIFF)
		base_dir, base_name = os.path.split(output_path)
		name_root, _ = os.path.splitext(base_name)

		png_path = os.path.join(base_dir, f"{name_root}.png")
		tiff_path = os.path.join(base_dir, f"{name_root}.tiff")

		plt.tight_layout()
		plt.savefig(
			png_path,
			dpi=PlotProperties.DPI,
			transparent=True,
			bbox_inches="tight",
			pad_inches=PlotProperties.PAD_INCHES,
		)
		plt.savefig(
			tiff_path,
			format="tiff",
			dpi=PlotProperties.DPI,
			bbox_inches="tight",
			pad_inches=PlotProperties.PAD_INCHES,
		)
		plt.close()
