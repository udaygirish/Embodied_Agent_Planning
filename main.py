import math
import os
import random
import sys
import argparse

import git
import imageio
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from RRT import*
from main_load_map import load_map
import cv2

import Navigation as nav
from Voice2Intent import Voice2Intent
from GetGoal import GoalPosition
from YOLO_OD import YoloDetection

TEST_PATH_ = "SCENES/"
TEST_PATH = None
scene_no = None
scene_name = False
test_scene = None
mp3d_scene_dataset = None

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Command Line')

    parser.add_argument('-test_path','--test_path',help='TEST PATH')

    parser.add_argument('-s' , '--scene_no',help='Enter scene no to load (1,6) to load default scenes')

    parser.add_argument('-scene' , '--scene' , help='Give path of the scene')

    args = parser.parse_args()

    test_path = args.test_path
    scene_no = args.scene_no
    scene = args.scene

    if test_path is not None :
        TEST_PATH = test_path
    else :
        TEST_PATH = TEST_PATH_

    if scene_no is None :
        scene_no = 1
    else :
        scene_no = int(scene_no)

display = True

## THE CODE STRUCTURE 

# ** STEP 1 **
# Separate and make a class for navigation intent part 
# which takes audio as input and outputs the intent
# Now call this function here which calls the fucntion to initiate
# taking speaker input and then give the intent

"""

"""

word_tokenizer_path = 'Speech2Intent/word_tokenizer_2.pkl'
unique_intent_path = 'Speech2Intent/unique_intent_2.pkl'
max_length_path = 'Speech2Intent/max_length_2.pkl'
model_path = "Speech2Intent/Chat_Model_2.h5"

vi = Voice2Intent(word_tokenizer_path,unique_intent_path,max_length_path,model_path)
vi.LoadData()
Intent = vi.Text2Intent()
# Intent = 'bed'


# ** STEP 2 **
# Initialize Randomisation of the agent 
# Try to find the goal object through RGB 
# Later this class shoudl add support to getting the depth information
# too ..  (Here separate the yolo inference function part too)
# Return Goal Point

TEST_PATH = f'{TEST_PATH}SCENE{scene_no}/'

output_path = "OUTPUTS/"

csv_path = 'top_down_map.csv'

if scene_no == 1 :
    test_scene = 'yPKGKBCyYx8.basis.glb'
    mp3d_scene_dataset = 'TEEsavR23oF.basis.json' #'yPKGKBCyYx8.basis.navmesh'
if scene_no == 2 :
    test_scene = 'sfbj7jspYWj.basis.glb'
    mp3d_scene_dataset = 'TEEsavR23oF.basis.json' #'sfbj7jspYWj.basis.navmesh'
if scene_no == 3 :
    test_scene = 'cVppJowrUqs.basis.glb'
    mp3d_scene_dataset = 'TEEsavR23oF.basis.json' #'cVppJowrUqs.basis.navmesh'
if scene_no == 4 :
    test_scene = 'oPj9qMxrDEa.basis.glb'
    mp3d_scene_dataset = 'TEEsavR23oF.basis.json' #'oPj9qMxrDEa.basis.navmesh'
if scene_no == 5 :
    test_scene = 'e3YKRHQRPNe.basis.glb'
    mp3d_scene_dataset = 'TEEsavR23oF.basis.json' #'e3YKRHQRPNe.basis.navmesh'
if scene_no == 6 :
    test_scene = 'Collierville.glb'
    mp3d_scene_dataset = 'Collierville.glb.json'

sim_settings = {
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "scene": TEST_PATH+test_scene,  # Scene path
    "scene_dataset": TEST_PATH+mp3d_scene_dataset,  # the scene dataset configuration files
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "depth_sensor": True,  # Depth sensor
    "semantic_sensor": True,  # Semantic sensor
    "seed": 8,  # used in the random navigation
    "enable_physics": True,  # kinematics only
}

meters_per_pixel = 0.025
    
agent_start_position = np.array([0  , 0.19205308 , 0])

gp = GoalPosition(TEST_PATH, test_scene , mp3d_scene_dataset, agent_start_position , sim_settings)

print(f"""
      
Simulator Settings : {gp.sim_settings}\n
Simulator SCENE : {gp.sim_settings['scene']}

""")

print(f'Searching for the {Intent} in the environment')

gp.run()
gp.make_agent()

Goal = gp.SpawnAtRandomPlaces(Intent)

gp.agent_state.position = gp.start_position
gp.agent.set_state(gp.agent_state)

position = gp.agent.get_state().position

# ** STEP 3 **
# Now take the goal point in 3D
# Convert to 2D GOal point in Top down map
# COnver tthe Random start point in 3D here also to 2D 
# If the random start point or 2D Point is not navigable return False

start_2D = nav.convert_points_to_topdown( gp.sim.pathfinder, gp.start_position, meters_per_pixel)
goal_2D = nav.convert_points_to_topdown(gp.sim.pathfinder, Goal[0], meters_per_pixel)
print("2D START: ", start_2D)
print("2D GOAL: ", goal_2D)

twoD_points = [start_2D, goal_2D]

# # ** STEP 4 **
# # Pass top down point to the RRT* or any other planner class
# # This will be a independent class out of this pipeline
# # This outputs the path points by takfing a top down map 
# # Retunr Path Points
path_points_2D = nav.get_path(gp.sim, TEST_PATH , csv_path , display, twoD_points)
print("PATH POINTS GIVEN BY RRT_STAR: ", path_points_2D)
print("LEN PATH POINTS GIVEN BY RRT_STAR: ", len(path_points_2D))
height = gp.sim.pathfinder.get_bounds()[0][1]

# # ** STEP 5**
# # Call a function which outputs the 3D parth points possibly interpolated 
# # by 10 times to get a smoother trajectory
path_points_3D = [nav.convert_topdown_to_points(gp.sim.pathfinder, points, height, meters_per_pixel) for points in path_points_2D]

print(f'Len points : Before (cleaning up) : {len(path_points_3D)}')
new_path_points_3D = []
for point in path_points_3D :
    if gp.sim.pathfinder.is_navigable(point) :
        new_path_points_3D.append(point)

path_points_3D = new_path_points_3D
print(f'Len points : After (cleaning up) : {len(path_points_3D)}')

print("PATH POINTS GIVEN BY 3D CONVERSION: ", path_points_3D)

# # ** STEP 6 **
# # RUN the Navigation using path points here and get every RGB and Depth 
# # Image now get the view and pass RGB and get the inference from the YOLO
# # Now overlay the output and save video

yd = YoloDetection()
observations_list = nav.get_sensory_observations(path_points_3D, output_path, gp.sim, gp.agent , yd , Intent , gp , scene_no)

"""
    The output will be saved to OUTPUTS/VIDEOS/
"""