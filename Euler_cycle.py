def neighbours_of_point(G,s):
    neighbours = []
    for i in range(len(G[s])):
        if G[s][i] >= 1:
            neighbours.append(i)
    return neighbours

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