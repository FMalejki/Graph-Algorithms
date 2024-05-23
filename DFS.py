def neighbours_of_point(G,s):
    neighbours = []
    for i in range(len(G[s])):
        if G[s][i] >= 1:
            neighbours.append(i)
    return neighbours

def DFS_algorithm(G):
    def DFS_visit(G,u):
        nonlocal time
        time += 1
        Visited[u] = True
        neighbours = neighbours_of_point(G,u)
        for neighbour in neighbours:
            if not Visited[neighbour]:
                Parent[neighbour] = u
                DFS_visit(G,neighbour)
        time += 1
    Visited = [False for _ in range(len(G))]
    Parent = [None for _ in range(len(G))]
    time = 0
    for i in range(len(G)):
        if not Visited[i]:
            DFS_visit(G,i)