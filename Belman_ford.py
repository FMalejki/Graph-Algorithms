import math
from queue import PriorityQueue

def neighbours_of_point(G,s):
    neighbours = []
    for i in range(len(G[s])):
        if G[s][i] >= 1:
            neighbours.append(i)
    return neighbours

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