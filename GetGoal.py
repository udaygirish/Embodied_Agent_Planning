import math
import os
import random
import sys

import git
import imageio
import magnum as mn
import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from YOLO_OD import YoloDetection
from Voice2Intent import Voice2Intent

class GoalPosition :
    def __init__(self,data_path,scene_name , json_name, robot_start_pos , sim_settings=None) :
        self.TEST_PATH = data_path
        self.test_scene = self.TEST_PATH + scene_name
        self.mp3d_scene_dataset = self.TEST_PATH + json_name
        if sim_settings is None :
            self.sim_settings = {
                "width": 512,  # Spatial resolution of the observations
                "height": 512,
                "scene": self.test_scene,  # Scene path
                "scene_dataset": self.mp3d_scene_dataset,  # the scene dataset configuration files
                "default_agent": 0,
                "sensor_height": 1.5,  # Height of sensors in meters
                "color_sensor": True,  # RGB sensor
                "depth_sensor": True,  # Depth sensor
                "semantic_sensor": True,  # Semantic sensor
                "seed": 1,  # used in the random navigation
                "enable_physics": True,  # kinematics only
            }
        else :
            self.sim_settings = sim_settings
        self.start_position = robot_start_pos

    def make_cfg(self,settings) :
        self.sim_cfg = habitat_sim.SimulatorConfiguration()
        self.sim_cfg.gpu_device_id = 0
        self.sim_cfg.scene_id = settings["scene"]
        self.sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
        self.sim_cfg.enable_physics = settings["enable_physics"]

        # Note: all sensors must have the same resolution
        sensor_specs = []

        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_leftl": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
            ),
            "turn_rightl": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
            )
        }

        return habitat_sim.Configuration(self.sim_cfg, [agent_cfg])

    def run(self) :
        self.cfg = self.make_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)
        self.action_names = list(self.cfg.agents[self.sim_settings["default_agent"]].action_space.keys())

    def make_agent(self) :
        self.agent = self.sim.initialize_agent(self.sim_settings["default_agent"])
        self.agent_state = habitat_sim.AgentState()
        self.agent_state.position = self.start_position


    def navigateAndSave(self,action=""):
        if action in self.action_names:
            observations = self.sim.step(action)
            # print("action: ", action)
            col_img = observations['color_sensor']
            dep_img = observations['depth_sensor']
            # print(col_img.shape , dep_img.shape)
            col_img = col_img[..., 0:3][..., ::-1]
            dep_img = np.clip(dep_img , 0, 10)/10.0
            dep_img = (dep_img * 255.0).astype(np.uint8)
            import cv2 as cv 
            dep_img = cv.merge([dep_img , dep_img , dep_img])
            # img = np.concatenate((col_img , dep_img) , axis=1)
            # print(view.shape)
            return col_img , dep_img

    def navigate(self,action) :
        observations = self.sim.step(action)
        col_img = observations['color_sensor']
        dep_img = observations['depth_sensor']
        # print(col_img.shape , dep_img.shape)
        col_img = col_img[..., 0:3][..., ::-1]
        dep_img = np.clip(dep_img , 0, 10)/10.0
        dep_img = (dep_img*255.).astype(np.uint8)
        import cv2 as cv 
        dep_img = cv.merge([dep_img , dep_img , dep_img])
        # img = np.concatenate((col_img , dep_img) , axis=1)
        # print(view.shape)
        return col_img,dep_img

    
    def SpawnAtRandomPlaces(self,Intent,num_places=20,actions=["turn_right","turn_left","move_forward"],save_images=True) :
        print('Spawning at random places ')
        self.actions = actions
        found = True
        count = 1
        boundaries = self.sim.pathfinder.get_bounds()
        min_bou , max_bou = boundaries[0] , boundaries[1]
        import cv2 as cv
        while found :
            sample = self.sim.pathfinder.get_random_navigable_point()
            if self.sim.pathfinder.is_navigable(sample) :
                self.agent_state.position = np.array(sample)
                self.agent.set_state(self.agent_state)
                for act in actions :
                    print(f'\r\tSample {count} : {sample} and action : {act}\t\t\t',end="")
                    # print(agent.get_state())
                    view , dep_view = self.navigateAndSave(act)
                    # path = Path + f'spawn_images/img_{count}_{act}.jpg'
                    # cv.imwrite(path,view)
                    YD = YoloDetection()
                    cv.imwrite(f'spawn_images/Img_{count}.jpg',view)
                    _ , view,point = YD.TestImg(view , Intent)
                    
                    if _ == True :
                        x,y,x2,y2 = point
                        self.act_at_goal = act
                        self.box_point = point
                        self.goal = sample
                        # dep_view_ = dep_view[y:y2,x:x2]
                        # avg = np.average(dep_view_)
                        path = f'spawn_images/img_OD_{count}.jpg'
                        
                        if save_images :
                            cv.imwrite(path,view)
                            cv.imwrite( f'spawn_images/img_dep_{count}.jpg', dep_view)
                        # _ = CheckDepth(view , dep_view)
                        print(f'\n {Intent} found at {sample}')
                        found = False
                        return sample,point
                count += 1
                if count-1 == num_places :
                    break
        print()
        return None,None


    def move2Obj(self,Goal_positoin=None,Intent=None) :
        self.agent_state.position = np.array(Goal_positoin[0])
        self.agent.set_state(self.agent_state)
        YD = YoloDetection()
        view = self.sim.get_sensor_observations()['color_sensor']
        dep_view = self.sim.get_sensor_observations()['depth_sensor']
        x,y,x2,y2 = self.box_point
        dep_view_ = dep_view[y:y2 , x:x2]
        avg = np.average(dep_view_)
        # path = f'spawn_images/img_OD_{count}.jpg'
        obj_path = []
        while avg > 25 :
            view,dep_view = self.navigate("move_forward")
            _ , view,point = YD.TestImg(view , Intent)
            if not _:
                break
            while 0.5*(point[0]+point[1]) <= 120 :
                    view,dep_view = self.navigate("turn_leftl")
                    _ , view,point = YD.TestImg(view , Intent)
            
            while 0.5*(point[0]+point[1]) >= 136:
                    view,dep_view = self.navigate("turn_rightl")
                    _ , view,point = YD.TestImg(view , Intent)
            # print(point)
            
            for i in range(2,4):
                if point[i]<=0:
                    point[i] = 1
            x,y,x2,y2 = point
            dep_view_ = dep_view[y:y2,x:x2]
            avg = np.average(dep_view_)
            point = self.agent.get_state().position
            obj_path.append(point)

        return obj_path

def main() :
    TEST_PATH = "/home/abven/habitat-sim/data/scene_datasets/habitat-test-scenes/"
    test_scene = 'TEEsavR23oF.basis.glb' 
    mp3d_scene_dataset = 'TEEsavR23oF.basis.json' 
    
    word_tokenizer_path = 'Speech2Intent/word_tokenizer_2.pkl'
    unique_intent_path = 'Speech2Intent/unique_intent_2.pkl'
    max_length_path = 'Speech2Intent/max_length_2.pkl'
    model_path = "Speech2Intent/Chat_Model_2.h5"
    # vi = Voice2Intent(word_tokenizer_path,unique_intent_path,max_length_path,model_path)
    # vi.LoadData()
    Intent = 'sofa' #vi.Text2Intent()

    gp = GoalPosition(TEST_PATH , test_scene , mp3d_scene_dataset)
    gp.run()
    gp.make_agent()

    Goal = gp.SpawnAtRandomPlaces(Intent)

    gp.agent_state.position = gp.start_position
    gp.agent.set_state(gp.agent_state)
    
    position = gp.agent.get_state().position
    if Goal is not None :
        print(f'Current position at : {position}')
        print(f'Goal is found at : {Goal}')

if __name__ == "__main__" :
    main()