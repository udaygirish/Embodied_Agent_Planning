import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import cKDTree
# from robot import*



class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.parent = None
        self.cost = 0.0

class RRT:
    def __init__(self, map_array, start, goal, n_configs, kdtree_d, radius):
        self.map_array = map_array
        self.size_row = map_array.shape[0]
        self.size_col = map_array.shape[1]
        
        self.start = Node(int(start[0]), int(start[1]))
        # print("START IN RRT: ", self.start.row)
        # print("START IN RRT: ", self.start.col)
        self.goal = Node(int(goal[0]), int(goal[1]))
    
        self.vertices = []
        self.n_configs = n_configs
        self.kdtree_d = kdtree_d
        self.radius = radius
        self.found = False
        

    def init_map(self):
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    def dis(self, node1, node2):
        return math.sqrt((node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2)

    def check_collision(self, node1, node2):
        x1, y1, x2, y2 = node1.row, node1.col, node2.row, node2.col
        map_array = self.map_array

        if (
            x1 < 0
            or x1 >= self.size_row
            or y1 < 0
            or y1 >= self.size_col
            or x2 < 0
            or x2 >= self.size_row
            or y2 < 0
            or y2 >= self.size_col
        ):
            return True

        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        steps = math.ceil(steps)
        obs_counter = 0
        if dx == 0 and dy == 0:
            return False

        for i in range(steps + 1):
            x, y = int(x1 + i * dx / steps), int(y1 + i * dy / steps)
            for w in range(int(x-self.radius), int(x+self.radius)):
                for h in range(int(y-self.radius), int(y+self.radius)):
                    if w < self.size_row and w > 0 and h < self.size_col and h > 0:
                        if map_array[w][h] == 1:
                            obs_counter +=1 
        # print("OBSTACLES:", obs_counter)
        if obs_counter < 1:
            return True
            # if map_array[x][y] == 1:   #CHNAGED
            #     return False
        # return True

    def get_new_point(self, goal_bias):
        x1, y1 = np.random.randint(0, self.size_row), np.random.randint(0, self.size_col)
        random_Node = Node(x1, y1)
        prob_point = np.random.choice([self.goal, random_Node], p=[goal_bias, 1 - goal_bias])
        return prob_point

    def get_nearest_node(self, point):
        min_d = math.inf
        for vertex in self.vertices:
            temp_dis = self.dis(vertex, point)
            if temp_dis < min_d:
                min_d = temp_dis
                nearest_node = vertex
        return nearest_node

    def extend(self, q_new, q_nearest):
        E_DIST = 8
        dist = self.dis(q_new, q_nearest)
        x_new, y_new, x_nearest, y_nearest = q_new.row, q_new.col, q_nearest.row, q_nearest.col

        dx, dy = x_new - x_nearest, y_new - y_nearest
        if dist != 0:
            x = x_nearest + dx * (E_DIST / dist)
            y = y_nearest + dy * (E_DIST / dist)

            if x < 0:
                x = 0
            elif x >= self.size_row:
                x = self.size_row - 1
            if y < 0:
                y = 0
            elif y >= self.size_col:
                y = self.size_col - 1

            q = Node(x, y)
            q.parent = q_nearest
            q.cost = q_nearest.cost + self.dis(q, q_nearest)
            return q
        else:
            return q_new

    def get_neighbors(self, new_node, neighbor_size):
        neighbor_list = []
        for vertex in self.vertices:
            temp_dis = self.dis(new_node, vertex)
            if temp_dis < neighbor_size:
                neighbor_list.append(vertex)
        return neighbor_list

    def get_path_points(self):
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker="o", color="y")
            plt.plot(
                [node.col, node.parent.col],
                [node.row, node.parent.row],
                color="y",
            )
        points = []
        
        if self.found:
            cur = self.goal
            points.append([self.goal.col, self.goal.row])
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot(
                    [cur.row, cur.parent.row],
                    [cur.col, cur.parent.col],
                    color="g",
                )
                # if cur.parent.col == self.start.
                points.append([int(cur.parent.col), int(cur.parent.row)]) # reversed it
                cur = cur.parent
                plt.plot(cur.row, cur.col, markersize=3, marker="o", color="b") # reverse printing
        points.reverse()
    
        plt.plot(self.start.row, self.start.col, markersize=6, marker="v", color="g")
        plt.plot(self.goal.row, self.goal.col, markersize=6, marker="v", color="r")
        
        # print("PATH POINTS BRUH: ", points)
        plt.show()
        
        # print("START POINT")

        return points
    
    def get_interpolated_points(self, node1, node2, num_points, continuous_path_points):
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        step_size = 1.0/num_points+1
        for i in range(1, num_points + 1):
            interpolated_points = [int(x1 + i*step_size*(x2 - x1)), int(y1 + i*step_size*(y2 - y1))]
            continuous_path_points.append(interpolated_points)

        # interpolated_points = [[int(x1 + i*step_size*(x2 - x1)), int(y1 + i*step_size*(y2 - y1))] for i in range(1, num_points + 1)] 
        return interpolated_points

    # def get_interpolated_points(self, node1 , node2 , num_points , continuous_path_points) :
    #     interpolated_points = []

    #     for i in range(num_points+1) :
    #         alpha = i/num_points
    #         point = (1-alpha)

    #     return interpolated_points
    
    def generate_continuos_points(self, points):
        continuous_path_points = []
        num_points = 5
        print("START NODE: ", points[len(points)-1])
        print("GOAL NODE: ", points[0])
        for i in range(len(points)-1):
            node1 = points[i]
            node2 = points[i+1] 
            continuous_path_points.append(node1)
            interpolated_points = self.get_interpolated_points(node1, node2, num_points, continuous_path_points) 
            # continuous_path_points.append(interpolated_points)
            continuous_path_points.append(node2)
        
        return continuous_path_points

    def RRT(self, n_pts=1000):
        self.init_map()

        goal_bias = 0.05

        for n in range(n_pts):
            q_new = self.get_new_point(goal_bias)
            q_nearest = self.get_nearest_node(q_new)
            q_new = self.extend(q_new, q_nearest)

            if self.check_collision(q_new, q_nearest) == True:
                q_new.parent = q_nearest
                q_new.cost = q_nearest.cost + self.dis(q_nearest, q_new)
                self.vertices.append(q_new)
                if q_new == self.goal:
                    self.found = True
                    break
            if (self.dis(q_new, self.goal) <= 20 and self.check_collision(q_new, self.goal) == True):
                self.found = True
                self.goal.parent = q_new
                self.goal.cost = q_new.cost + self.dis(q_new, self.goal)
                break

        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
            self.draw_map()
        else:
            print("No path found")

        


class RRTStar(RRT):
    def __init__(self, map_array, start, goal, n_configs, kdtree_d, radius):
        super().__init__(map_array, start, goal, n_configs, kdtree_d, radius)

    def extend(self, q_new, q_nearest):
        E_DIST = 20
        dist = self.dis(q_new, q_nearest)
        x_new, y_new, x_nearest, y_nearest = q_new.row, q_new.col, q_nearest.row, q_nearest.col

        dx, dy = x_new - x_nearest, y_new - y_nearest
        if dist != 0:
            x = x_nearest + dx * (E_DIST / dist)
            y = y_nearest + dy * (E_DIST / dist)

            if x < 0:
                x = 0
            elif x >= self.size_row:
                x = self.size_row - 1
            if y < 0:
                y = 0
            elif y >= self.size_col:
                y = self.size_col - 1

            q = Node(x, y)
            q.parent = q_nearest
            q.cost = q_nearest.cost + self.dis(q, q_nearest)

            if self.dis(q, self.goal) <= E_DIST:
                q.parent = q_nearest
                q.cost = q_nearest.cost + self.dis(q_nearest, self.goal)
                return q

            # Check for collisions during extension
            if self.check_collision(q, q_nearest):
                neighbors = self.get_neighbors(q, E_DIST)
                for neighbor in neighbors:
                    if (
                        self.check_collision(neighbor, q)
                        and neighbor.cost + self.dis(neighbor, q) < q.cost
                    ):
                        q.parent = neighbor
                        q.cost = neighbor.cost + self.dis(neighbor, q)

                return q
            else:
                return q_nearest
        else:
            return q_new

    

    def RRT_star(self, n_pts=2000):
        self.init_map()

        goal_bias = 0.05

        for n in range(n_pts):
            q_new = self.get_new_point(goal_bias)
            q_nearest = self.get_nearest_node(q_new)
            q_new = self.extend(q_new, q_nearest)

            if self.check_collision(q_new, q_nearest):
                q_new.parent = q_nearest
                q_new.cost = q_nearest.cost + self.dis(q_nearest, q_new)
                neighbors = self.get_neighbors(q_new, 15)
                self.vertices.append(q_new)

                for neighbor in neighbors:
                    if (
                        self.check_collision(neighbor, q_new)
                        and neighbor.cost + self.dis(neighbor, q_new) < q_new.cost):
                        q_new.parent = neighbor
                        q_new.cost = neighbor.cost + self.dis(neighbor, q_new)

        # Connect the goal node to the tree
        q_goal_nearest = self.get_nearest_node(self.goal)
        self.goal.parent = q_goal_nearest
        self.goal.cost = q_goal_nearest.cost + self.dis(q_goal_nearest, self.goal)

        # Rewire the tree to find shorter paths
        for q_node in self.vertices:
            neighbors = self.get_neighbors(q_node, 15)
            for neighbor in neighbors:
                if ( self.check_collision(neighbor, q_node)
                    and q_node.cost + self.dis(q_node, neighbor) < neighbor.cost):
                    neighbor.parent = q_node
                    neighbor.cost = q_node.cost + self.dis(q_node, neighbor)
                    # self.vertices.append(self.goal)
                    self.found = True

        # points = []  
        # for vertex in self.vertices:
        #     tuple(vertex)
        #     points.append(tuple)
       
        if self.found == True:
            self.vertices.append(self.goal)
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
            # self.draw_map()
            points = self.get_path_points()
            print("RRT_STAR PATH POINTS", points)
            # continuous_path_points = self.generate_continuos_points(points)
            # print("CONTINUOUS PATH POINTS:", continuous_path_points)
            return points
        else: 
            print("NO PATH FOUND!")
            return []
        
        
        
        




# class InformedRRTStar(RRTStar):
#     def __init__(self, map_array, start, goal, n_configs, kdtree_d):
#         super().__init__(map_array, start, goal, n_configs, kdtree_d)
#         self.kdtree = None

#     def update_kdtree(self):
#         self.kdtree = cKDTree([node.col for node in self.vertices])

#     def informed_extend(self, q_new, q_nearest):
#         E_DIST = 15
#         dist = self.dis(q_new, q_nearest)
#         x_new, y_new, x_nearest, y_nearest = q_new.row, q_new.col, q_nearest.row, q_nearest.col

#         dx, dy = x_new - x_nearest, y_new - y_nearest
#         if dist != 0:
#             x = x_nearest + dx * (E_DIST / dist)
#             y = y_nearest + dy * (E_DIST / dist)

#             if x < 0:
#                 x = 0
#             elif x >= self.size_row:
#                 x = self.size_row - 1
#             if y < 0:
#                 y = 0
#             elif y >= self.size_col:
#                 y = self.size_col - 1

#             q = Node(x, y)
#             q.parent = q_nearest
#             q.cost = q_nearest.cost + self.dis(q, q_nearest)
#             neighbors = self.get_neighbors(q, E_DIST)
#             for neighbor in neighbors:
#                 if (
#                     self.check_collision(neighbor, q)
#                     and neighbor.cost + self.dis(neighbor, q) < q.cost
#                 ):
#                     q.parent = neighbor
#                     q.cost = neighbor.cost + self.dis(neighbor, q)

#             return q
#         else:
#             return q_new

#     def informed_RRT_star(self, n_pts=1000):
#         self.init_map()

#         goal_bias = 0.05

#         for n in range(n_pts):
#             q_new = self.get_new_point(goal_bias)
#             q_nearest = self.get_nearest_node(q_new)
#             q_new = self.informed_extend(q_new, q_nearest)

#             if self.check_collision(q_new, q_nearest) == True:
#                 q_new.parent = q_nearest
#                 q_new.cost = q_nearest.cost + self.dis(q_nearest, q_new)
#                 neighbors = self.get_neighbors(q_new, 15)
#                 self.vertices.append(q_new)
#                 for neighbor in neighbors:
#                     if (
#                         self.check_collision(neighbor, q_new)
#                         and neighbor.cost + self.dis(neighbor, q_new) < q_new.cost
#                     ):
#                         q_new.parent = neighbor
#                         q_new.cost = neighbor.cost + self.dis(neighbor, q_new)

#         self.goal.parent = min(
#             self.vertices,
#             key=lambda vertex: vertex.cost + self.dis(vertex, self.goal),
#         )
#         self.found = True

#         steps = len(self.vertices) - 2
#         length = self.goal.cost
#         print("It took %d nodes to find the current path" % steps)
#         print("The path length is %.2f" % length)

#         self.draw_map()

# Example Usage:
# map_array = np.zeros((100, 100))
# start = (10, 10)
# goal = (90, 90)

# rrt = RRT(map_array, start, goal, "random", 1000, 2)
# rrt.RRT()

# rrt_star = RRTStar(map_array, start, goal, "random", 1000, 2)
# rrt_star.RRT_star()

# informed_rrt_star = InformedRRTStar(map_array, start, goal, "random", 1000, 2)
# informed_rrt_star.update_kdtree()
# informed_rrt_star.informed_RRT_star()