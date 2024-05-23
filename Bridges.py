def neighbours_of_point(G,s):
    neighbours = []
    for i in range(len(G[s])):
        if G[s][i] >= 1:
            neighbours.append(i)
    return neighbours

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