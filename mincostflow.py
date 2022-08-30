import heapq

INF = 1e9

def minCostFlow(g, flow):
    vertexSize = len(g)
    res = 0
    h = [0] * vertexSize
    prevv = [0] * vertexSize
    preve = [0] * vertexSize
    
    while(flow > 0):
        q = []
        d = [INF] * vertexSize
        d[0] = 0
        heapq.heappush(q, (0, 0))
        
        while(len(q) != 0):
            p = heapq.heappop(q)
            v = p[1]
            if(d[v] < p[0]): continue
            
            for i in range(len(g[v])):
                e: Edge = g[v][i]
                if(e.cap > 0 and d[e.to] > d[v] + e.cost + h[v] - h[e.to]):
                    d[e.to] = d[v] + e.cost + h[v] - h[e.to]
                    prevv[e.to] = v
                    preve[e.to] = i
                    heapq.heappush(q, (d[e.to], e.to))
        
        if(d[vertexSize - 1] == INF):
            return g, -1
        
        for v in range(vertexSize):
            h[v] += d[v]
        
        df = flow
        vi = vertexSize - 1
        while(vi != 0):
            df = min(df, g[prevv[vi]][preve[vi]].cap)
            vi = prevv[vi]
        
        flow -= df
        res += df * h[vertexSize - 1]
        
        vi = vertexSize - 1
        while(vi != 0):
            e1: Edge = g[prevv[vi]][preve[vi]]
            e1.cap -= df
            g[prevv[vi]][preve[vi]] = e1
            
            e2: Edge = g[vi][e1.rev]
            e2.cap += df
            g[vi][e1.rev] = e2
            
            vi = prevv[vi]
    return g, res

class Edge:
    def __init__(self, to, cap, cost, rev):
        self.to = to
        self.cap = cap
        self.cost = cost
        self.rev = rev

V, E, F = map(int, input().split())
g = [[] for _ in range(V)]
for i in range(E):
    u, v, c, d = map(int, input().split())
    g[u].append(Edge(v, c, d, len(g[v])))
    g[v].append(Edge(u, 0, -d, len(g[u]) - 1))

_, res = minCostFlow(g, F)
print(res)
