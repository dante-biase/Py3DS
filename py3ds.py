import os
import time

import numpy as np
import open3d as o3d
from open3d.visualization import gui
from matplotlib import pyplot as plt
from colour import Color


def new_cloud():
	return o3d.geometry.PointCloud()


def load_cloud(cloud_file):
	return o3d.io.read_point_cloud(cloud_file)


def save_cloud(cloud, to_dir, name="cloud"):
	output_path = os.path.join(to_dir, name + ".pcd")
	o3d.io.write_point_cloud(output_path, cloud)
	return output_path


def copy(geometry):
	return type(geometry)(geometry)


def visualize(geometries, size=(1920, 1080)):

	o3d.visualization.draw_geometries(
		geometries,
		window_name="Open3D Visualizer",
		width=size[0], height=size[1],
	)

	# app = gui.Application.instance
	# app.initialize()
	# vis = o3d.visualization.O3DVisualizer(title="Open3D Visualizer", width=size[0], height=size[1])
	#
	# for i, geometry in enumerate(geometries):
	# 	vis.add_geometry(str(i), geometry)
	#
	# app.add_window(vis)
	# app.run()

	# vis = o3d.visualization.VisualizerWithEditing()
	# vis.create_window()
	# vis.add_geometry(geometries[0])
	# while True:
	# 	vis.poll_events()
	# 	vis.update_renderer()


def capture_scene(geometries, vvs_path, output_path, background_color=Color("white"), show_progress=False):
	# setup visualizer
	vis = o3d.visualization.Visualizer()

	# set background color
	# opt = vis.get_render_option()
	# opt.background_color = np.asarray(background_color.rgb)

	vis.create_window(visible=show_progress)
	for geometry in geometries:
		vis.add_geometry(geometry)

	# read camera params
	view_settings = o3d.io.read_pinhole_camera_parameters(vvs_path)
	ctr = vis.get_view_control()
	ctr.convert_from_pinhole_camera_parameters(view_settings)

	# updates
	for geometry in geometries:
		vis.update_geometry(geometry)
	vis.poll_events()
	vis.update_renderer()

	# capture image
	time.sleep(1)
	vis.capture_screen_image(output_path)

	# close
	vis.destroy_window()


def spherical_segmentation(cloud, center, radius):
	points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
	distances = np.linalg.norm(points - np.array(center), axis=1)

	inlier_cloud, outlier_cloud = new_cloud(), new_cloud()
	inlier_cloud.points = o3d.utility.Vector3dVector(points[distances <= radius])
	inlier_cloud.colors = o3d.utility.Vector3dVector(colors[distances <= radius])
	outlier_cloud.points = o3d.utility.Vector3dVector(points[distances > radius])
	outlier_cloud.colors = o3d.utility.Vector3dVector(colors[distances > radius])

	return inlier_cloud, outlier_cloud


def planar_segmentation(cloud, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
	plane_model, inliers = cloud.segment_plane(
		distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
	)
	inlier_cloud = cloud.select_by_index(inliers)
	outlier_cloud = cloud.select_by_index(inliers, invert=True)
	return inlier_cloud, outlier_cloud


def dbscan_segmentation(cloud, eps=0.02, min_points=10, print_progress=False):
	def get_color_id(c):
		return int(''.join([str(v).replace('.', '', 1) for v in c]))

	labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
	max_label = labels.max()
		
	points = np.asarray(cloud.points)
	colors = np.asarray(cloud.colors)

	noise = new_cloud()
	noise.points = o3d.utility.Vector3dVector(points[labels < 0])
	noise.colors = o3d.utility.Vector3dVector(colors[labels < 0])
	#noise.paint_uniform_color([0, 0, 0])

	points = points[labels >= 0]
	colors = colors[labels >= 0]

	cluster_colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))[:, :3]
	cluster_colors = cluster_colors[labels >= 0]
	colors_ids = np.array([get_color_id(c) for c in cluster_colors])

	clusters = []
	for cc in np.unique(cluster_colors, axis=0):
		cc_id = get_color_id(cc)
		cluster = new_cloud()
		cluster.points = o3d.utility.Vector3dVector(points[colors_ids == cc_id])
		cluster.colors = o3d.utility.Vector3dVector(colors[colors_ids == cc_id])
		clusters.append(cluster)

	return clusters, noise


def merge(clouds):
	combined_cloud = new_cloud()
	for cloud in clouds:
		combined_cloud += cloud

	return combined_cloud


def create_bounding_box(cloud):
	return cloud.get_axis_aligned_bounding_box()


def paint(geometry, color):
	geometry_painted = copy(geometry)
	if type(geometry) != o3d.geometry.PointCloud:
		geometry_painted.color = color.rgb
		return geometry_painted
	else:
		cloud_painted = copy(geometry)
		cloud_painted.paint_uniform_color(color.rgb)
		return cloud_painted
