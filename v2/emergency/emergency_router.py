import heapq
import numpy as np

class EmergencyRouter:
    def dijkstra(self, adj_matrix, start, goal):
        n = len(adj_matrix)
        distances = {i: float('inf') for i in range(n)}
        distances[start] = 0
        pq = [(0, start)]
        parents = {start: None}
        
        while pq:
            curr_dist, curr = heapq.heappop(pq)
            if curr == goal: break
            if curr_dist > distances[curr]: continue
            
            for neighbor in range(n):
                if adj_matrix[curr][neighbor] > 0:
                    weight = adj_matrix[curr][neighbor]
                    dist = curr_dist + weight
                    if dist < distances[neighbor]:
                        distances[neighbor] = dist
                        parents[neighbor] = curr
                        heapq.heappush(pq, (dist, neighbor))
                        
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parents.get(curr)
        return path[::-1], distances[goal]
        
    def a_star(self, adj_matrix, coords, start, goal):
        # Coordinates (x,y) for heuristic
        def heuristic(u, v):
            return np.sqrt((coords[u][0]-coords[v][0])**2 + (coords[u][1]-coords[v][1])**2)
            
        n = len(adj_matrix)
        g_score = {i: float('inf') for i in range(n)}
        g_score[start] = 0
        f_score = {i: float('inf') for i in range(n)}
        f_score[start] = heuristic(start, goal)
        pq = [(f_score[start], start)]
        parents = {start: None}
        
        while pq:
            _, curr = heapq.heappop(pq)
            if curr == goal: break
            
            for neighbor in range(n):
                if adj_matrix[curr][neighbor] > 0:
                    tentative_g = g_score[curr] + adj_matrix[curr][neighbor]
                    if tentative_g < g_score[neighbor]:
                        parents[neighbor] = curr
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(pq, (f_score[neighbor], neighbor))
                        
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parents.get(curr)
        return path[::-1], g_score[goal]
