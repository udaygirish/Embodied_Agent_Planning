import math
import os
import random
import sys

import git
import imageio
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from main_load_map import load_map

import cv2

from RRT import*
import astar_search as ma

def get_reachability(interpolated_node_3D, height, pathfinder, meters_per_pixel):
    print("Is point navigable? " + str(pathfinder.is_navigable(interpolated_node_3D)))
    max_search_radius = 5.0
    print("Interpolated 3D points: ", interpolated_node_3D)
    # print("Distance to obstacle: " + str(pathfinder.distance_to_closest_obstacle(interpolated_node_3D, max_search_radius)))
    hit_record = pathfinder.closest_obstacle_surface_point(interpolated_node_3D, max_search_radius)
    if math.isinf(hit_record.hit_dist):
        print("No obstacle found within search radius.")
        interpolated_node_2D = convert_points_to_topdown(pathfinder,interpolated_node_3D, meters_per_pixel)
        return interpolated_node_2D
    else:
    # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
        perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
        print("Perturbed point : " + str(perturbed_point))
        print("Is point navigable? " + str(pathfinder.is_navigable(perturbed_point)))
        snapped_point = pathfinder.snap_point(perturbed_point)
        print("Snapped point : " + str(snapped_point))
        print("Is point navigable? " + str(pathfinder.is_navigable(snapped_point)))
        interpolated_node_2D = convert_points_to_topdown(pathfinder, snapped_point, meters_per_pixel)
    return interpolated_node_2D

def generate_continuos_path(node1, node2, height, num_points, continuous_path_nodes, pathfinder, meters_per_pixel):
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        step_size = 1.0/num_points+1
        for i in range(0, num_points + 2):
            interpolated_node = [int(x1 + i*step_size*(x2 - x1)), int(y1 + i*step_size*(y2 - y1))]
            interpolated_node_3D = convert_topdown_to_points(pathfinder, interpolated_node, height, meters_per_pixel )
            interpolated_node_2D = get_reachability(interpolated_node_3D, height, pathfinder, meters_per_pixel)
            continuous_path_nodes.append(interpolated_node_2D)
        return continuous_path_nodes

def get_interpolation(points_2D, height, pathfinder, meters_per_pixel):
    continuous_path_nodes = []
    num_points = 5
    for i in range(len(points_2D)-1):
        node1 = points_2D[i]
        node2 = points_2D[i+1] 
        continuous_path_nodes = generate_continuos_path(node1, node2, height, num_points, continuous_path_nodes, pathfinder, meters_per_pixel)
       
    return continuous_path_nodes

def convert_points_to_topdown(pathfinder, points, meters_per_pixel):    #Function to convert 3D points to 2D
    # points_topdown = []
    bounds = pathfinder.get_bounds()
    # for point in points:
        # convert 3D x,z to topdown x,y
    px = (points[0] - bounds[0][0]) / meters_per_pixel
    py = (points[2] - bounds[0][2]) / meters_per_pixel
    points_topdown = [px,py]

    return points_topdown

def convert_topdown_to_points(pathfinder, points, height, meters_per_pixel):   #Function to convert 2D points to 3D
    bounds = pathfinder.get_bounds()
    px = points[1] 
    # print("PX:", px)
    # print("METERS PER PIXEL: ", meters_per_pixel)
    point1_3D = px*meters_per_pixel + bounds[0][0]
    point2_3D = -height
    py = points[0] 
    point3_3D = py*meters_per_pixel + bounds[0][2]
    points_3D = np.array([point1_3D, point2_3D, point3_3D], dtype= np.float32)

    return points_3D


# display a topdown map with matplotlib
def display_map(topdown_map, output_path, point1=None, point2=None, key_points=None):
    if point1 is None and point2 is None and key_points is None:
        img = Image.fromarray(topdown_map)
        img.save("top_down_map.png")

    if point1 is not None and point2 is not None:
        print("REACHING HERE")
        plt.plot(point1, point2, marker="v", markersize=20, alpha=0.8)
        plt.savefig(output_path +"Map_with_start_and_goal.png", bbox_inches='tight', pad_inches=0, transparent=True)
    # plot points on map
    if key_points is not None:
        # print("REACHING HERE")
        for point in key_points:
            plt.plot(point[0], point[1], marker="v", markersize=20, alpha=0.5)    
        plt.savefig(output_path + "Map_key_points_overlay.png",bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.show(block=False)

def linear_interpolation(node1, node2, num_points, meters_per_pixel):
    path = []
    for i in range(num_points + 1):
        alpha = i / num_points
        interpolated_point = (1 - alpha) * np.array(node1) + alpha * np.array(node2)
        path.append(interpolated_point * meters_per_pixel)
    return path

def spline_interpolation(nodes, num_points):
    from scipy.interpolate import CubicSpline
    t = np.linspace(0, 1, len(nodes))
    cs = CubicSpline(t, nodes, bc_type='natural')
    path = cs(np.linspace(0, 1, num_points))
    return path

def get_path(sim, output_path, csv_path ,display, twoD_points,num_points=5):
    if display:
        from habitat.utils.visualizations import maps
    meters_per_pixel = 0.025  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
    found_path = False
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        csv_file_path = output_path +  csv_path
        radius = 1.5

        """
            RRT_Planner Path Generating
        """
        # environment_map_array = load_map(csv_file_path=csv_file_path)
        # RRT_planner = RRTStar(environment_map_array, twoD_points[0], twoD_points[1], 1200 , 10,radius=radius )
        # path_points_2D = RRT_planner.RRT_star()

        """
            AStar Path Generating -> Takes csv path and start and goal positions
        """
        astar_path = ma.run(csv_file_path,[int(twoD_points[0][0]) , int(twoD_points[0][1])], [int(twoD_points[1][0]) , int(twoD_points[1][1])])
        # ma.draw_path(environment_map_array , astar_path)
        path_points_2D = astar_path


        height = sim.pathfinder.get_bounds()[0][1]

        new_path = []
        for path_i in range(len(path_points_2D)-1) :
            path_points = linear_interpolation([path_points_2D[path_i]], [path_points_2D[path_i+1]], num_points, 1)
            # path_points = spline_interpolation(sim,[[path_points_2D[path_i]] , [path_points_2D[path_i+1]]] , 15)
            path_points = [i[0].tolist() for i in path_points]
            path_points = [[j,i] for i,j in path_points]
            new_path.extend(path_points)
            
        path_points = new_path # The path from starting position to the goal
        if path_points != []:
            found_path = True

        if found_path:
            meters_per_pixel = 0.025
            scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min
            if display:
                # print("REACHING HERE")
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
                display_map(topdown_map=top_down_map,output_path='spawn_images/')
                recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
                top_down_map = recolor_map[top_down_map]
                grid_dimensions = top_down_map.shape
                print("Dimensions of Map:", grid_dimensions)

                grid_tangent = mn.Vector2(
                    path_points[1][1] - path_points[0][1], path_points[1][0] - path_points[0][0]
                )
                # grid_tangent = mn.Vector2(
                #     path_points[1][1] - path_points[0][1], path_points[1][0] - path_points[0][0]
                # )
                path_initial_tangent = grid_tangent / grid_tangent.length()
                initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                # draw the agent and path_points on the map
                
                # print("NEW TRAJECTROY: ", path_points_modified)
                # maps.draw_path(top_down_map, path_points)
                # # maps.draw_path(top_down_map, path_points)
                # maps.draw_agent(
                #     top_down_map, path_points[0], initial_angle, agent_radius_px=8
                # )
                # point1 = path_points[0]
                # point2 = path_points[len(path_points)-1]                
                # display_map(top_down_map,output_path, point1=point1, point2=point2)
                # map_filename = output_path + "Map_with_agent_path_overlay.png"
                # imageio.imsave(map_filename, top_down_map)

    return path_points

def display_sample(ix, output_path, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        # img_to_video(depth_img, "depth")
        depth_frame = np.array(depth_img)
        # print(depth_frame.shape)
        # vid_depth.write(depth_frame)
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plt.imshow(data)
        save_sample(output_path, data, ix)
       
    plt.show(block=False)
    
def get_sensory_observations(path_points_3D, output_path, sim, agent , yd , Intent , gp , scene_no):
   
    display_path_agent_renders = True
    observations_list = []
    from datetime import datetime
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

    output_filename = f'{output_path}/VIDEOS/{scene_no}_output_RGB_video_{current_datetime}.avi'
    RGB_vid_write = cv2.VideoWriter(output_filename,  fourcc, 60, (gp.sim_settings['width'] , gp.sim_settings['height']) , isColor=True )
    output_filename = f'{output_path}/VIDEOS/{scene_no}_output_DEP_video_{current_datetime}.avi'
    DEP_vid_write = cv2.VideoWriter(output_filename,  fourcc, 60, (gp.sim_settings['width'] , gp.sim_settings['height']) , isColor=True )
    # output_filename = f'{output_path}/VIDEOS/{scene_no}_output_SEM_video_{current_datetime}.avi'
    # SEM_vid_write = cv2.VideoWriter(output_filename,  fourcc, 30, (gp.sim_settings['width'] , gp.sim_settings['height']) , isColor=True )

    if display_path_agent_renders:
        print("Rendering observations at path points:")

        agent_state = habitat_sim.AgentState()

        Len_P_3d = len(path_points_3D)
        goa = False
        for ix, point in enumerate(path_points_3D):
            if ix < len(path_points_3D) - 1:
                print(f'\r \t {ix+1}/{Len_P_3d} \t\t' ,end="")

                if point[0] == gp.goal[0] and abs(gp.goal[2]-point[2]) < .5 :
                    agent_state.position = point
                    goa = True
                else :
                    tangent = path_points_3D[ix + 1] - point
                    agent_state.position = point
                    tangent_orientation_matrix = mn.Matrix4.look_at(
                        point, point + tangent, np.array([0, 1.0 , 0])
                    )
                    tangent_orientation_q = mn.Quaternion.from_matrix(
                        tangent_orientation_matrix.rotation()
                    )
                    agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    
                try:
                    agent.set_state(agent_state)
                    if goa :
                        view , dep_view = gp.navigateAndSave(gp.act_at_goal)
                        _ , view , p_ = yd.TestImg(view , Intent)
                        RGB_vid_write.write(view)
                        DEP_vid_write.write(dep_view)
                        break
                    observations = sim.get_sensor_observations()
                    rgb = observations["color_sensor"]
                    depth = observations["depth_sensor"]
                    
                    observations_list.append(observations) # Appending every frame (both rgb and depth images)

                    depth = np.clip(depth, 0, 10)/ 10.0

                    depth = (depth * 255.).astype(np.uint8)
                    depth = cv2.merge([depth , depth , depth])
                    # print(sem)
                    # sem = cv2.merge([sem , sem , sem])
                    
                    if len(rgb.shape) > 2:
                        rgb = rgb[..., 0:3][..., ::-1]

                    """
                        To see object detections in the frames
                    """

                    # _ ,img,p_ = yd.TestImg(rgb,Intent)
                    # if _ :
                    #     rgb = img
                    # view = np.concatenate((rgb,depth) , axis=1)

                    RGB_vid_write.write(rgb)
                    DEP_vid_write.write(depth)
                    # SEM_vid_write.write(semantic)

                    # save_sample(output_path,rgb,ix) # To save frame as an image
                except Exception as e:
                    # print("ERROR: ", e)
                    continue
    
    print()

    """
        Move near to the object from goal position
    """

    RGB_vid_write.release()
    DEP_vid_write.release()
    # SEM_vid_write.release()
    return observations_list

def save_sample(output_path, data, ix):
    cv2.imwrite(output_path+'IMAGES/'+str(ix)+".png" , data)

def get_video(observations_list, output_path, show_video):
    vut.make_video(
        observations=observations_list,
        primary_obs="color_sensor",
        primary_obs_type='color',
        video_file= output_path + "continuous_path",
        fps=3,
        open_vid=False,
    )



