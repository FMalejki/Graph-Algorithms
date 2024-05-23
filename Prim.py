import math
from queue import PriorityQueue

def neighbours_of_point(G,s):
    neighbours = []
    for i in range(len(G[s])):
        if G[s][i] >= 1:
            neighbours.append(i)
    return neighbours

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