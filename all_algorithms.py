#IMPORTANT NOTES:
#Get-Content filename.py | Set-Clipboard

#BFS algorithm:
#a wave that goes thru whole graph searching for particular element
#usage:
#1. shortest path - when graph has no weights
#2. cohesion - if we dont visit any of the nodes
#3. detecting cycles - if we visit an already visited node that is not our parent
#4. bicoherence - coloring graphs - or even lenght
#complexity of BFS algorithm:
#list rep. O(V+E)
#matrix rep. O(V^2)
import queue
import math

class vertex:
    def __init__(self,visited,distance,parent):
        self.visited = False
        self.distance = -1
        self.parent = None

def BFS_algorithm(G,s = 0):
    Q = queue.Queue()
    #also for list implementation this part will strongly differ
    Visited = [False for _ in range(len(G))]
    Distance = [-1 for _ in range(len(G))]
    Parent = [None for _ in range(len(G))]
    #s could be a random number or every number
    Visited[s] = True
    Distance[s] = 0
    #parent dont needs to be actualised
    Q.put(s)
    while not Q.empty():
        u = Q.get()
        #for every neighbour of a vertex - this part will acctually differ
        neighbours = neighbours_of_point(G,u)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                Visited[neighbour] = True
                Distance[neighbour] = Distance[u] + 1
                Parent[neighbour] = u
                Q.put(neighbour)

def neighbours_of_point(G,s):
    #this is the part that is different in list or matrix implementation
    #in matrix:
    neighbours = []
    for i in range(len(G[s])):
        if G[s][i] >= 1:
            neighbours.append(i)
    return neighbours

    #G variable is a matrix containing of data
    """
    0  1  2  3 
   0 [0,1,1,0],
   1 [1,0,1,1],
   2 [1,1,0,0],
   3 [0,1,0,0]
   
   something simmiliar to this 
    """

G = [[0,1,1,0],
   [1,0,1,1],
   [1,1,0,0],
   [0,1,0,0]]

#end
#DFS algorithm:
#basically recursion with reversion
#complexity:
#list rep. O(V+E)
#matrix rep. O(V^2)

#usages:
#cohesion - if the function launches second time recurssion in first for in major function it is not consistent
#bicoherence - coloring
#cycles detection - same in BFS
#topological sorting - only for directed graphs
#strongly coherent components - only for directed graphs
#Euler cycle
#bridges

def DFS_algorithm(G):
    def DFS_visit(G,u):
        nonlocal time
        time += 1
        #time of visit
        Visited[u] = True
        #function the same as in BFS
        #different when different data type used
        neighbours = neighbours_of_point(G,u)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                Parent[neighbour] = u
                DFS_visit(G,neighbour)
        time += 1
        #time of execution

    #for matrix implementation (the same as BFS)
    #this will differ for list implementation
    Visited = [False for _ in range(len(G))]
    Parent = [None for _ in range(len(G))]
    time = 0
    for i in range(len(G)):
        #without this if statement the DFS_visit would execute for every node - even if its visited
        if not Visited[i]:
            #if a graph is consistent this statement would be executed only once
            #the DFS_visit function is executed from here second time we know
            #that the graph is not coherent
            DFS_visit(G,i)

#Topological sort of DAG graph (Direct Acyclic Graph)
#using DFS algorithm
#in function DFS_visit after for loop (when a vertex is fully processed) we need to add this vertex to a stack
#or just append and reverse a table (!!!)

G1 = [[0,1,1,0],
      [0,0,1,1],
      [0,0,0,0],
      [0,0,0,0]]

def Topological_cycle_algorithm(G1):
    def DFS_visit(G1,u):
        nonlocal time
        time += 1
        #time of visit
        Visited[u] = True
        #function the same as in BFS
        #different when different data type used
        neighbours = neighbours_of_point(G1,u)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                Parent[neighbour] = u
                DFS_visit(G1,neighbour)
        time += 1
        Result.append(u)
        #time of execution

    #for matrix implementation (the same as BFS)
    #this will differ for list implementation
    Visited = [False for _ in range(len(G1))]
    Parent = [None for _ in range(len(G1))]
    Result = []
    time = 0
    for i in range(len(G1)):
        #without this if statement the DFS_visit would execute for every node - even if its visited
        if not Visited[i]:
            #if a graph is consistent this statement would be executed only once
            #the DFS_visit function is executed from here second time we know
            #that the graph is not coherent
            DFS_visit(G1,i)
    return Result[::-1]


#Finding Euler cycle
#a cycle that goes thru every vertex in graph
#using DFS algorithm we modify it to permit visiting the same vertex multiple times but instead of visited
#table for vertexes we make simmilar table for graph edges - they cant be visited twice
#in function DFS_visit after for loop (when a vertex is fully processed) we need to add this vertex to a stack
#or just append and reverse a table (!!!)

def Finding_euler_cycle_algorithm(G2):
    def DFS_visit(G2,u):
        neighbours = neighbours_of_point(G2,u)
        for neighbour in neighbours:
            #this implementation differs when graph is Direct or not
            if not Visited[u][neighbour] or not Visited[neighbour][u]:
                Visited[u][neighbour] = True
                Visited[neighbour][u] = True
                #there wouldnt be this line
                DFS_visit(G2, neighbour)
                Result.append(u)


    Visited = [[False for _ in range(len(G2))] for _ in range(len(G2))]
    Result = []
    #we basicly know that a graph we are checking has an euler cycle
    #so we dont need to make for loop in case it wasnt coherent
    Result.append(0)
    DFS_visit(G2,0)
    return Result[::-1]

G2 = [
    [0,0,0,1,1],
    [0,0,1,1,0],
    [0,1,0,1,0],
    [1,1,1,0,1],
    [1,0,0,1,0]
]

#Strongly coherent graph components
#if we want how many graph has coherent components we need to use modified DFS
#Steps:
#1. run a DFS Algorithm memorising time of finalizing operations on vertexes
#2. reverse direction of graph
#3. execute DFS second time on reversed direction graph and in descending order of time
#   the vertexes were processed

G3 = [
    [0,1,0,0,0,0,0,1,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0],
    [1,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,0,0,1],
    [0,0,0,0,0,0,0,0,1,0,0],

]

def Strongly_coherent(G):
    def DFS_visit(G,u):
        nonlocal time
        Visited[u] = True
        neighbours = neighbours_of_point(G,u)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                DFS_visit(G,neighbour)
        time+=1
        Time[u] = time

    def reverse_graph(G):
        #this function will strongly differ when the data
        #structure instead of matrix would be list
        for i in range(len(G)):
            for j in range(i,len(G)):
                G[i][j], G[j][i] = G[j][i], G[i][j]
        return G

    def DFS_second(G,i):
        Visited[i] = True
        neighbours = neighbours_of_point(G, i)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                DFS_second(G, neighbour)

    def has_time_other_attr_than_inf(Time):
        contains_other_than_inf = any(x != -1 for x in Time)
        return contains_other_than_inf

    #First DFS:
    Visited = [False for _ in range(len(G))]
    Time = [0 for _ in range(len(G))]
    time = 0
    for i in range(len(G)):
        if not Visited[i]:
            DFS_visit(G,i)

    G = reverse_graph(G)

    #Second DFS
    Visited = [False for _ in range(len(G))]
    counter = 0
    while has_time_other_attr_than_inf(Time):
        MAX = max(Time)
        index_of_max = Time.index(MAX)
        Time[index_of_max] = -1
        #this could be done better by modifying Visited array to hold time
        if not Visited[index_of_max]:
            DFS_second(G,index_of_max)
            counter += 1
    return counter

#Bridges in undirected graphs
#if an edge is not on any graph cycle its a bridge
#Steps:
#1. Execute DFS memorising for every vertex its time of visit (NOT PROCESSING AS IN PREVIOUS!!!)
#2. Calculate low(v) for every vertex
#3. Bridges are vertexes ((v,p(v)) where d(v) (time of visit) = low(v)

#low definition:
#the minimum of the time value and the minimum of the neighbors to
#which the backward edges lead (unused but unusable) and the minimum
#of the recursive DFS return
#not checked but on paper works

def Bridges_algorithm(G):
    def DFS_visit(G,u):
        nonlocal time
        time += 1
        Time[u] = time
        Low[u] = time
        #time of visit
        Visited[u] = True
        #function the same as in BFS
        #different when different data type used
        neighbours = neighbours_of_point(G,u)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                Parent[neighbour] = u
                DFS_visit(G,neighbour)
        for neighbour in neighbours:
            if Low[neighbour] < Low[u] and Parent[u] != neighbour:
                Low[u] = Low[neighbour]
        #time of execution

    #for matrix implementation (the same as BFS)
    #this will differ for list implementation
    Visited = [False for _ in range(len(G))]
    Parent = [None for _ in range(len(G))]
    Time = [0 for _ in range(len(G))]
    Low = [0 for _ in range(len(G))]
    time = 0
    for i in range(len(G)):
        #without this if statement the DFS_visit would execute for every node - even if its visited
        if not Visited[i]:
            #if a graph is consistent this statement would be executed only once
            #the DFS_visit function is executed from here second time we know
            #that the graph is not coherent
            DFS_visit(G,i)

#Finding shortest paths in graph
#data structure of graph:
#1. matrix
#2. list - with additional data about weight of every edge

#ways to solve problem
#BFS with a counter of weight of every edge
#DJIKSTRA - jumps to vertexes without counting, the weights can be a float num but unsigned
#priotity queue
#1. initialise a priority queue with all edges with value infiity (minimum type)
#2. change start.distance = 0
#3. until a queue isnt empty get element with the lowest value
#4. for every edge coming out of the edge with the lowest value make a relaxation
from queue import PriorityQueue

G4 = [
    [0, 4, 1, 0],
    [4, 0, 0, 1],
    [1, 2, 0, 5],
    [0, 1, 5, 0]
]

#complexity O(ElogV)
#MAKE A DIJKSTRA ALGORITHM WITH HEAPQ NOT WITH PRIORITY QUEUE

def Dijkstra_algorithm(G,s=0):
    def Relax(vertex, neighbour):
        #this will differ with other data struct
        if Distance[neighbour] > Distance[vertex[1]] + G[vertex[1]][neighbour]:
            Distance[neighbour] = Distance[vertex[1]] + G[vertex[1]][neighbour]
            Parent[neighbour] = vertex[1]
            priority_queue.put((Distance[neighbour],neighbour))

    priority_queue = PriorityQueue()
    #first is DISTANCE second is NUMBER OF VERTEX
    starting_vertex = (0,s)
    Distance = [math.inf for _ in range(len(G))]
    Distance[s] = 0
    Parent = [None for _ in range(len(G))]
    Taken = [False for _ in range(len(G))]
    priority_queue.put(starting_vertex)
    while not priority_queue.empty():
        smallest = priority_queue.get()
        if Taken[smallest[1]]:
            continue
        neighbours = neighbours_of_point(G,smallest[1])
        Taken[smallest[1]] = True
        for neighbour in neighbours:
            Relax(smallest, neighbour)

    print(Distance)

def Dijkstra_algorithm_without_priority_que(G, s = 0):
    def Relax(vertex, neighbour):
        if Distance[neighbour] > Result[vertex] + G[vertex][neighbour]:
            Distance[neighbour] = Result[vertex] + G[vertex][neighbour]
            Parent[neighbour] = vertex
    Distance = [math.inf for _ in range(len(G))]
    Result = [-1 for _ in range(len(G))]
    Distance[s] = 0
    Parent = [None for _ in range(len(G))]
    #assuming that graph is cohese
    while any(x == -1 for x in Result):
        MIN = min(Distance)
        index = Distance.index(MIN)
        Result[index] = MIN
        Distance[index] = math.inf
        neighbours = neighbours_of_point(G,index)
        for neighbour in neighbours:
            if Result[neighbour] == -1:
                Relax(index,neighbour)
    print(Result)
#Dijkstra_algorithm(G4)
#Dijkstra_algorithm_without_priority_que(G4)

#Searching for the lowest path in Graph when the edges are negative too
#Floyd - Warshall algorithm
#Steps:
#1. initialisation
#2. Relax for every vertex (V - 1 times)
#3. Verification - for every edges check if v.d <= u.d + w(u,v)
#   if not there is a cycle with negative weight

def Belman_Ford_algorithm(G,s = 0):
    def Relax(vertex, neighbour):
        #this will differ with other data struct
        if Distances[neighbour] > Distances[vertex] + G[vertex][neighbour]:
            Distances[neighbour] = Distances[vertex] + G[vertex][neighbour]
            Parent[neighbour] = vertex

    Distances = [math.inf for _ in range(len(G))]
    Parent = [None for _ in range(len(G))]
    Distances[s] = 0
    for _ in range(len(G)-1):
        for i in range(len(G)):
            for j in range(len(G)):
                if G[i][j] != 0:
                    Relax(i,j)

    #check for negative cycles
    for i in range(len(G)):
        for j in range(len(G)):
            #if Distances[i] + G[i][j] >= Distances[j] everything is alright
            if G[i][j] != 0 and Distances[i] + G[i][j] < Distances[j]:
                return None
    return Distances

#Specialised algorithms:
#Floyd-Warshall
#complexity - O(V^3)

def Floyd_Warshall_algorithm(G):
    for k in range(len(G)):
        for i in range(len(G)):
            for j in range(len(G)):
                G[i][j] = min(G[i][j],G[i][k]+G[k][j])

#Floyd - Warshall algorithm for all to all relations
def Floyd_Warshall_algorithm_ALL(graph):
    V = len(graph)
    dist = [[float('inf')] * V for _ in range(V)]

    # Initialize distances
    for i in range(V):
        for j in range(V):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j] != 0:
                dist[i][j] = graph[i][j]

    # Floyd-Warshall algorithm
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    print(dist)
    return dist

#it returns all-all relation


#MINIMAL SPANDING TREE PROBLEM
#we consider only graphs with certain characteristics
#1. consistent
#2. weighted
#3. undirected

#coditions:
#A is a subset of MST of a graph
#e is a candidate to add to A
#e needs to fit certain conditions:
#1. e is not in A
#2. A + e has got no cycle
#3. e weight is minimal from edges fitting those conditions

#Kruskal algorithm
#1. Sort vertexes by weights
#2. Make a list A
#3. Analise vertexes in ascending (not descending) direction
#4. if A + e dont consist of a cycle add e to A
#5. return A

#Checking if a new A subset consist of a cycle:
#Data structure FindUnion

class Node:
    def __init__(self,value):
        self.val = value
        self.parent = self
        self.rank = 0

#FindUnion Compression

def find(x):
    if x.parent != x:
        x.parent = find(x.parent)
    return x.parent

#FindUnion Connecting

def union(x,y):
    x = find(x)
    y = find(y)
    if x == y:
        return
    if x.rank > y.rank:
        y.parent = x
    else:
        x.parent = y
        if x.rank == y.rank:
            y.rank += 1

G6 = [
    [0, 2, 0, 6, 0],
    [2, 0, 3, 8, 5],
    [0, 3, 0, 0, 7],
    [6, 8, 0, 0, 9],
    [0, 5, 7, 9, 0]
]

def Kruskal_algorithm(G):
    # Step 1: Initialize nodes
    nodes = [Node(i) for i in range(len(G))]
    # i is basicaly a number of node - name of the node
    # value of it isnt used anywhere

    # Step 2: Create an edge list
    #it will differ when graph representation is other than matrix
    edges = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):  # To avoid duplicate edges
            if G[i][j] > 0:
                edges.append((G[i][j], i, j))
    #here we are making an list of all edges in order to sort then
    #as in the algorithm

    # Step 3: Sort edges by weight
    edges.sort()

    # Step 4: Process edges
    #A its a table just like in algorithm
    A = []
    #edges is a (weight, i, j) struct so we are collecting those for every iteration
    for weight, i, j in edges:
        if find(nodes[i]) != find(nodes[j]):
            #if they are not going to make a cycle
            #it literally checks if those nodes are in the same set if not
            #they wont make a cycle in any case
            union(nodes[i], nodes[j])
            #then we merge them - we know that this is perfectly safe
            A.append((i, j, weight))
            #we add those edges to A list like in algorithm

    # Step 5: Output the MST edges
    """
    print("Edges in the MST:")
    for u, v, weight in A:
        print(f"Edge ({u}, {v}) with weight {weight}")
    """

#Prim algorithm

#Steps:
#1. Make a Dijkstra algorithm with some modifications

#modfications:
#1. do not sum weights - just take them without summing with previous ones

G5 = [
    [0,1,0,0,0,3],
    [1,0,2,0,4,0],
    [0,2,0,1,0,1],
    [0,0,1,0,3,0],
    [0,4,0,3,0,2],
    [3,0,1,0,2,0]
]

def Prim_algorithm(G,s=0):
    def Relax(vertex, neighbour):
        #this will differ with other data struct
        if not Taken[neighbour] and Distance[neighbour] >= G[vertex[1]][neighbour]:
            Distance[neighbour] = G[vertex[1]][neighbour]
            Parent[neighbour] = vertex[1]
            priority_queue.put((Distance[neighbour],neighbour))

    priority_queue = PriorityQueue()
    #first is DISTANCE second is NUMBER OF VERTEX
    starting_vertex = (0,s)
    Distance = [math.inf for _ in range(len(G))]
    Distance[s] = 0
    Parent = [None for _ in range(len(G))]
    Taken = [False for _ in range(len(G))]
    priority_queue.put(starting_vertex)
    while not priority_queue.empty():
        smallest = priority_queue.get()
        if Taken[smallest[1]]:
            continue
        neighbours = neighbours_of_point(G,smallest[1])
        Taken[smallest[1]] = True
        for neighbour in neighbours:
            Relax(smallest, neighbour)
    print(Parent)

