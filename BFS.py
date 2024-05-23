import queue

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