from PIL import Image
import numpy as np
from RRT import *
import pandas as pd
# from PRM import PRM



def load_map(csv_file_path):
    """Load map from an image and return a 2D binary numpy array
    where 0 represents obstacles and 1 represents free space
    """
    # Load the image with grayscale
    # img = Image.open(file_path).convert("L")
    # # # Rescale the image
    # size_x, size_y = img.size
    # new_x, new_y = int(size_x * resolution_scale), int(
    #     size_y * resolution_scale
    # )
    # img = img.resize((new_x, new_y), Image.LANCZOS)

    # map_array = np.asarray(img, dtype="uint8")

    # # # Get binry image
    # threshold = 128
    # map_array = 1 * (map_array > threshold)
    df = pd.read_csv(csv_file_path, header=None)

# Convert DataFrame to NumPy array
    map_array = df.to_numpy()

    # Result 2D numpy array
    return map_array


if __name__ == "__main__":
    # Load the map

    csv_file_path = "/home/pradnya/Documents/MP_Project/habitat/habitat-sim/examples/tutorials/Experiments/binary_top_down_map.csv"

    map_array = load_map(csv_file_path)
    print('map',map_array.shape) 
   
    
    # if map_array[173][610] == 0:
    #     print("TRUE")   
    # if map_array[139][267] == 0:
    #     print("TRUE") 
    # Planningclass
    radius = 1.5
    # PRM_planner = PRM(map_array, radius)
    # # RRT_planner = RRT(map_array, start, goal)

    # # robot = CircularRobot()
    start = [610, 173]
    goal = [267, 139]
    # # # method = RRT(sampling_method = 'RRT_star', n_configs = 300, kdtree_d=10)
    # rrt = RRT(map_array, start, goal, 1500, 10, 1.5)
    # rrt.RRT()
    # # robot = CircularRobot(radius=1.5)
    rrt_star = RRTStar(map_array, start, goal, 1200, 10, 1.5)
    rrt_star.RRT_star()
    # # Search with PRM
    # PRM_planner.sample(n_pts=1000, sampling_method="uniform")
    # PRM_planner.search(start, goal)
    # PRM_planner.sample(n_pts=2000, sampling_method="gaussian")
    # PRM_planner.search(start, goal)
    # PRM_planner.sample(n_pts=20000, sampling_method="bridge")
    # PRM_planner.search(start, goal)

    # new sampler with a different bridge concept
    # PRM_planner.sample(n_pts=3000, sampling_method='new_sampler')
    # PRM_planner.search(start,goal)

    # # new sampler with weighted concept
    # PRM_planner.sample(n_pts=3000, sampling_method='weighted_sampler')
    # PRM_planner.search(start,goal)

    # PRM_planner.incremental_sample_prm(n_pts=1000, sampling_method='uniform', new_samples=300)

    # Search with RRT
    # RRT_planner.RRT(n_pts=1000)

    # PRM_planner.benchmarker(start, goal)



    


