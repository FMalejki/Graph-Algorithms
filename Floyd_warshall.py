def Floyd_Warshall_algorithm(G):
    for k in range(len(G)):
        for i in range(len(G)):
            for j in range(len(G)):
                G[i][j] = min(G[i][j],G[i][k]+G[k][j])

#infinity where there is no connection between
#edges - zero on diagonal - number when there is
#a weighted connection