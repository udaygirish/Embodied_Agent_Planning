# Basic searching algorithms
from util import Stack, Queue, PriorityQueueWithFunction
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import csv
from queue import PriorityQueue

class Node():

    def __init__(self, position = None , parent=None):
        self.parent = parent # Parent for the node
        self.position = position # current position of node

        self.g = 0 # current g score of the node
        self.h = 0 # current h score of the node
        self.f = 0 # current f score of the node => f = g + n


def possible_moves(grid,a,visited,alg=None,goal=None) :
    x,y = a
    if alg is None :
        moves = [[0,1] , [1,0] , [0,-1] , [-1,0]] #[-1,0] , [0,-1] , [0,1] , [1,0]] # UP , DOWN , LEFT , RIGHT -> the directions it can move
    else :
        moves = [[0,1] , [1,0] , [0,-1] , [-1,0]][::-1]
    poss_moves = []

    for i,j in moves :
        if 0<=x+i<len(grid) and 0<=y+j<len(grid[0]) and grid[x+i][y+j] == 0 and not visited[x+i][y+j] :
            poss_moves.append([x+i,y+j])
    return poss_moves

def path_Finder(parent, goal):
    path = []
    row , col = goal
    while parent[row][col] is not None:
        path.append([row , col])
        row, col = parent[row][col]
    path.append([row, col])
    return path[::-1]

def bfs(grid, start, goal):
    '''Return a path found by BFS alogirhm 
       and th`e number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]. 
            If no path exists return an empty list [].
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('maps/test_map.csv')
    >>> bfs_path, bfs_steps = bfs(grid, start, goal)
    It takes 10 steps to find a path using BFS
    >>> bfs_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False


    grid[goal[0]][goal[1]] = 0 # If goal place is fixed with 1 then need to replace with 0 .

    rows = len(grid)
    cols = len(grid[0])
    
    visited_blocks = [[False for _ in range(cols)] for _ in range(rows)] # Making every block is not visited
    parent = [[None for _ in range(cols)] for _ in range(rows)] # Making None for every block (not as a parent)

    queue = Queue() # Using given util Package's Queue class
    queue.push([start[0],start[1]])
    # print(grid,visited_blocks , start[0],start[1])
    visited_blocks[start[0]][start[1]] = True
    
    while not queue.isEmpty() :
        cur_x, cur_y = queue.pop() # Removing first item from the queue
        moves = possible_moves(grid,[cur_x,cur_y],visited_blocks) # Finding possible moves in all 4 directions
        steps += 1 # Visited nodes
        if [cur_x,cur_y] == goal :
            found = True
            break

        for i, j in moves :
            queue.push([i, j])
            visited_blocks[i][j] = True
            parent[i][j] = [cur_x,cur_y] # assigning cur_x and cur_y as a parent for i,j
            
    path = path_Finder(parent,goal)

    if found:
        print(f"It takes {steps} steps to find a path using BFS")
    else:
        path = []
        print("No path found")
    return path , steps

def dfs(grid, start, goal):
    '''Return a path found by DFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]. If no path exists return
            an empty list []
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('maps/test_map.csv')
    >>> dfs_path, dfs_steps = dfs(grid, start, goal)
    It takes 9 steps to find a path using DFS
    >>> dfs_path
    [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    grid[goal[0]][goal[1]] = 0 # If goal place is fixed with 1 then need to replace with 0 .

    rows = len(grid)
    cols = len(grid[0])

    stack = Stack()
    Explored = Stack() 

    # The possible movements -> W , N , S , E => (-1,0) , (0,-1) , (0,1) , (1,0)
    # parent = [[None for _ in range(cols)] for _ in range(rows)]
    
    visited_blocks = [[False for _ in range(cols)] for _ in range(rows)]
    # visited_blocks[start[0]][start[1]] = True

    stack.push(start)

    while not stack.isEmpty() :
        out = stack.pop()
        moves = possible_moves(grid,out,visited_blocks,alg='dfs')
        Explored.push(out)

        visited_blocks[start[0]][start[1]] = True
        steps += 1

        if goal == out :
            found = True
            path.append(goal)
            break

        for i,j in moves :
            # parent[i][j] = [out[0],out[1]]
            Explored.push([i,j])
            stack.push([i,j])
            visited_blocks[i][j] = True
            # print(i,j,Explored)
            # stack.traverse()
            # Explored.traverse()
        if out not in path :
            path.append(out)

    if not found :
        path = []

    if found:
        print(f"It takes {steps} steps to find a path using DFS")
    else:
        path = []
        print("No path found")
    return path, steps

def heuristic_distance(current_node,goal,type='manhattan_distance') :

    if type == "manhattan_distance" :
        return abs(current_node[0] - goal[0]) + abs(current_node[1] - goal[1])

    if type == "manhattan_distance_directional" :
        return 4*(abs(current_node[0] - goal[0]) + abs(current_node[1] - goal[1]))

    if type == "euclidean_distance" :
        return np.sqrt((current_node[0] - goal[0])**2 + (current_node[1] - goal[1])**2)

    if type == "euclidean_distance_directional" :
        return 4*(np.sqrt((current_node[0] - goal[0])**2 + (current_node[1] - goal[1])**2))

    if type == "euclidean_distance_directional_diagonal" :
        return 8*(np.sqrt((current_node[0] - goal[0])**2 + (current_node[1] - goal[1])**2))

    if type == "chebyshev_distance" :
        return max(abs(current_node[0] - goal[0]) , abs(current_node[1] - goal[1]))

# def 

def astar(grid, start, goal):
    '''Return a path found by A* alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]. If no path exists return
            an empty list []
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('maps/test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    grid = np.array(grid).T

    rows = len(grid)
    cols = len(grid[0])

    # print(grid)

    grid[goal[0]][goal[1]] = 0 # If goal place is fixed with 1 then need to replace with 0 .

    g_score = [[np.inf for _ in range(cols)] for _ in range(rows)]
    # h_score = [[np.inf for _ in range(rows)] for _ in range(cols)]
    f_score = [[np.inf for _ in range(cols)] for _ in range(rows)]

    g_score[start[0]][start[1]] = 0

    """
        Need to set the type in heuristic distance .

    """

    f_score[start[0]][start[1]] = heuristic_distance(start,goal,type='manhattan_distance')

    open = PriorityQueue()
    """
        pushing the F_value , h_value , position into the open PriorityQueue .
    """
    open.put((heuristic_distance(start,goal)+g_score[start[0]][start[1]],
              heuristic_distance(start,goal),
              start))

    aPath={}
    visited_blocks = [[False for _ in range(cols)] for _ in range(rows)]

    # print(open.get())

    while not open.empty() :

        current_node = open.get()[2] # Current node position .
        steps += 1

        if current_node==goal :
            found = True
            break

        moves = possible_moves(grid,current_node,visited_blocks)

        for i,j in moves :
            cell = (i,j)
            temp_g_score = g_score[current_node[0]][current_node[1]]+1
            temp_f_score = temp_g_score + heuristic_distance([i,j],goal) 

            if temp_f_score < f_score[i][j] :
                g_score[i][j] = temp_g_score
                f_score[i][j] = temp_f_score
                open.put((temp_f_score,heuristic_distance([i,j],goal),[i,j]))
                aPath[cell] = tuple(current_node)
                

    fpath = {}
    node = tuple(goal)
    path.append(goal)

    while node!=tuple(start) and found :
        fpath[aPath[node]]=node
        node = aPath[node]
        path.append(list(node))
    path = path[::-1]
    
    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        path = []
        print("No path found")
    return path, steps

# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()
