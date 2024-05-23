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
