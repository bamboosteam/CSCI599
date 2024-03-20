import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest
import heapq


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    
    Overall process:
    Reference: https://github.com/mikedh/trimesh/blob/main/trimesh/remesh.py#L207
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.
    
    # The following implementation considers only the interior cases
    # You should also consider the boundary cases and more iterations in your submission
    """
    
    def single_subdivision(mesh):
        # prepare geometry for the loop subdivision
        vertices, faces = mesh.vertices, mesh.faces # [N_vertices, 3] [N_faces, 3]
        edges, edges_face = faces_to_edges(faces, return_index=True) # [N_edges, 2], [N_edges]
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)
        
        # split edges to interior edges and boundary edges
        edge_inter = np.sort(grouping.group_rows(edges, require_count=2), axis=1)
        edge_bound = grouping.group_rows(edges, require_count=1)
        
        # set also the mask for interior edges and boundary edges
        edge_bound_mask = np.zeros(len(edges), dtype=bool)
        edge_bound_mask[edge_bound] = True
        edge_bound_mask = edge_bound_mask[unique]
        edge_inter_mask = ~edge_bound_mask
        
        ###########
        # Step 1: #
        ###########
        # Calculate odd vertices to the middle of each edge.
        odd = vertices[edges[unique]].mean(axis=1) # [N_oddvertices, 3]
        
        # connect the odd vertices with even vertices
        # however, the odd vertices need further updates over it's position
        # we therefore complete this step later afterwards.
        
        ###########
        # Step 2: #
        ###########
        # find v0, v1, v2, v3 and each odd vertex
        # v0 and v1 are at the end of the edge where the generated odd vertex on
        # locate the edge first
        e = edges[unique[edge_inter_mask]]
        # locate the endpoints for each edge
        e_v0 = vertices[e][:, 0]
        e_v1 = vertices[e][:, 1]
        
        # v2 and v3 are at the farmost position of the two triangle
        # locate the two triangle face
        edge_pair = np.zeros(len(edges)).astype(int)
        edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
        edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
        opposite_face1 = edges_face[unique]
        opposite_face2 = edges_face[edge_pair[unique]]
        # locate the corresponding edge
        e_f0 = faces[opposite_face1[edge_inter_mask]]
        e_f1 = faces[opposite_face2[edge_inter_mask]]
        # locate the vertex index and vertex location
        e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
        e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
        e_v2 = vertices[e_v2_idx]
        e_v3 = vertices[e_v3_idx]
        
        # update the odd vertices based the v0, v1, v2, v3, based the following:
        # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
        odd[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0
        
        ###########
        # Step 3: #
        ###########
        # find vertex neightbors for even vertices and update accordingly
        neighbors = graph.neighbors(edges=edges[unique], max_index=len(vertices))
        # convert list type of array into a fixed-shaped numpy array (set -1 to empties)
        neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
        # if the neighbor has -1 index, its point is (0, 0, 0), so that it is not included in the summation of neighbors when calculating the even
        vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
        # number of neighbors
        k = (neighbors + 1).astype(bool).sum(axis=1)
        
        # calculate even vertices for the interior case
        beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
        even = (
            beta[:, None] * vertices_[neighbors].sum(1)
            + (1 - k[:, None] * beta[:, None]) * vertices
        )

        # calculate even vertices for the boundary case
        if edge_bound_mask.any():
            # find the boundary vertices
            bound_edges = np.unique(edges[unique][edge_bound_mask])
            vert_bound_mask = np.zeros(len(vertices), dtype=bool)
            vert_bound_mask[bound_edges] = True

            # update the even vertices for the boundary case
            # find the immediate adjacent vertices from the boundary vertices
            boundary_neighbors = neighbors[vert_bound_mask]
            for i, nei in enumerate(boundary_neighbors):
                for j, n in enumerate(nei):
                    if n not in bound_edges:
                        nei[j] = -1
                boundary_neighbors[i] = nei

            even[vert_bound_mask] = 0.75 * vertices[vert_bound_mask] + 0.125 * vertices_[boundary_neighbors].sum(axis=1)
            
        
        ############
        # Step 1+: #
        ############
        # complete the subdivision by updating the vertex list and face list
        
        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack(
            [
                faces[:, 0],
                odd_idx[:, 0],
                odd_idx[:, 2],
                odd_idx[:, 0],
                faces[:, 1],
                odd_idx[:, 1],
                odd_idx[:, 2],
                odd_idx[:, 1],
                faces[:, 2],
                odd_idx[:, 0],
                odd_idx[:, 1],
                odd_idx[:, 2],
            ]
        ).reshape((-1, 3)) # [N_face*4, 3]

        # stack the new even vertices and odd vertices
        new_vertices = np.vstack((even, odd)) # [N_vertex+N_edge, 3]
        
        return trimesh.Trimesh(new_vertices, new_faces)
    
    for _ in range(iterations):
        mesh = single_subdivision(mesh)
    
    return mesh

def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    def get_normal(face):
        v0, v1, v2 = face
        e1, e2 = vertices[v1] - vertices[v0], vertices[v2] - vertices[v0]
        return np.cross(e1, e2)

    def get_k(face):
        normal = get_normal(face)
        d = np.dot(normal, vertices[face[0]])
        arr = np.append(normal, d)
        return np.outer(arr, arr)

    def get_cost(edge):
        v0, v1 = edge
        q0, q1 = vert_quadric_errors[v0], vert_quadric_errors[v1]
        v0, v1 = vertices[v0], vertices[v1]
        q = q0 + q1

        q[-1] = np.array([0, 0, 0, 1])
        if np.linalg.det(q) == 0:
            x = (v0 + v1) / 2
            x = np.append(x, 1)
        else:
            x = np.dot(np.linalg.inv(q), np.array([0, 0, 0, 1]))
        return x, x.T @ (q0 + q1) @ x

        # b, w = q[:3, :3], q[:3, 3]
        # if np.linalg.det(b) == 0:
        #     x = (v0 + v1) / 2
        # else:
        #     x = np.dot(np.linalg.inv(b), w)
        # x = np.append(x, 1)
        # return x, x.T @ q @ x
    

    vertices, faces, normals = mesh.vertices, mesh.faces, np.copy(mesh.face_normals)
    face_adjacency = np.copy(mesh.face_adjacency)
    vertices_deleted, faces_deleted = np.zeros(len(vertices), dtype=bool), np.zeros(len(faces), dtype=bool)

    # calculate the quadric error for each vertex
    # compute the plane equation for each face and sum the quadric error for each vertex
    
    dot_product = np.einsum('ij,ij->i', vertices[faces[:, 0]], normals) # N_faces: calculate d
    arr_ = np.concatenate([normals, dot_product[:, np.newaxis]], axis=1) # N_faces, 4: calculate [a, b, c, d]
    k = np.einsum('ij,ik->ijk', arr_, arr_) # N_faces, 4, 4: calculate k for each face
    k_ = np.vstack((k, np.zeros((1, 4, 4)))) # N_faces+1, 4, 4: add a zero matrix at the end for the new vertex
    incident_faces = np.copy(mesh.vertex_faces) # N_vertices, N_incident_faces: incident faces for each vertex
    vert_quadric_errors = k_[incident_faces].sum(1) # N_vertices, 4, 4: calculate the quadric error for each vertex

    # compute cost for each edge
    edges, edges_face = faces_to_edges(faces, return_index=True) # N_edges, 2: edges, N_edges: face index for each edge
    edges.sort(axis=1) # sort the edges
    unique, inverse = grouping.unique_rows(edges) # N_unique_edges, 2: unique edges, N_edges: inverse index for unique edges
    heap = [] # heap for the cost
    merged_points = [] # merged points for the new vertices

    for i, edge in enumerate(edges):
        new_point, cost = get_cost(edge)
        heap.append((cost, i, i))
        merged_points.append(new_point)
    
    heapq.heapify(heap)

    # incremental collapse until reaching the target face count
    face_num = len(faces)
    while heap and face_num > face_count:
        cost, edge_idx, new_point_idx = heapq.heappop(heap)
        v0, v1 = edges[edge_idx]
        if v0 == v1 or vertices_deleted[v0] or vertices_deleted[v1]:
            continue

        # collapse the edge edge_idx
        # find the faces to remove
        f0 = edges_face[edge_idx]
        neighbors = np.where(face_adjacency == f0)[0]
        v0, v1 = edges[edge_idx]

        for nei in neighbors:
            f1 = face_adjacency[nei][0] if face_adjacency[nei][0] != f0 else face_adjacency[nei][1]
            if v0 in faces[f1] and v1 in faces[f1]: # both vertices are in the face f1, this is the face to remove
                break
        
        # if either of f or f1 is already deleted, invalid so skip it
        if faces_deleted[f0] or faces_deleted[f1]:
            continue

        neighbor_edges_f0 = np.where(edges_face == f0)[0]
        neighbor_edges_f1 = np.where(edges_face == f1)[0]
        
        # Remove vertices and faces by changing the flags
        vertices_deleted[v1] = True
        faces_deleted[f0], faces_deleted[f1] = True, True
        face_num -= 2 if f0 != f1 else 1

        # update the vertices
        vertices[v0] = merged_points[new_point_idx][:3]

        # update from v1 to v0 in the faces and edges as reconnection
        v1_faces =  np.where(np.isin(faces, v1))[0]

        for i in v1_faces:
            face = faces[i]
            if faces_deleted[i]: 
                continue
            if v1 in face and v0 in face:
                faces_deleted[i] = True
                continue
            elif v1 in face:
                for j in range(3):
                    if face[j] == v1:
                        face[j] = v0
                        break
        v1_edges =  np.where(np.isin(edges, v1))[0]

        for i in v1_edges:
            edge = edges[i]
            if edge[0] == v1 and edge[1] == v0:
                continue
            elif edge[0] == v1:
                edges[i] = np.array([edge[1], v0])
            else:
                edges[i] = np.array([edge[0], v0])

        # update edges_face, face_adjacency
        f0_neighbor_faces = np.delete(neighbors, np.where(neighbors == f0))
        f1_neighbors = np.where(face_adjacency == f1)[0]
        f1_neighbor_faces = np.delete(f1_neighbors, np.where(f1_neighbors == f1))

        for i in neighbor_edges_f0:
            if edges_face[i]  == f0:
                edges_face[i] = edges_face[f0_neighbor_faces[0]]
        
        for j in neighbor_edges_f1:
            if edges_face[j] == f1:
                edges_face[j] = edges_face[f1_neighbor_faces[0]]
        
        for i in neighbors:
            if face_adjacency[i][0] == f0:
                face_adjacency[i][0] = edges_face[f0_neighbor_faces[0]]
            else:
                face_adjacency[i][1] = edges_face[f0_neighbor_faces[0]]
        
        for i in f1_neighbors:
            if face_adjacency[i][0] == f1:
                face_adjacency[i][0] = edges_face[f1_neighbor_faces[0]]
            else:
                face_adjacency[i][1] = edges_face[f1_neighbor_faces[0]]

        # recompute the quadric error for the vertices
        neighbor_faces = np.where(np.isin(faces, v0))[0]
        for i in neighbor_faces:
            if faces_deleted[i]:
                k[i] = np.zeros((4, 4))
                continue
            k[i] = get_k(faces[i])
        k_ = np.vstack((k, np.zeros((1, 4, 4))))
        for i in neighbor_faces:
            vert_quadric_errors[faces[i]] = k_[faces[i]].sum(0)
        # vert_quadric_errors[v0] = k_[neighbor_faces].sum(0)

        # update the cost for the edges
        v0_edges = np.where(np.isin(edges, v0))[0]
        for i, elm in enumerate(heap):
            _, e, point_idx = elm
            if e in v0_edges:
                # recalculate the cost
                new_point, cost = get_cost(edges[e])
                merged_points[point_idx] = new_point
                heap[i] = (cost, e, point_idx)
        
        heapq.heapify(heap)
    
    # recreate the mesh
    return trimesh.Trimesh(vertices, faces[~faces_deleted])

if __name__ == '__main__':
    # Load mesh and print information
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')
    
    # # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=6)
    
    # # TODO: implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    # # # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('../writeup/cube_subdivided1.obj')
    
    # # quadratic error mesh decimation
    # # mesh_decimated = mesh.simplify_quadric_decimation(8)
    
    # # TODO: implement your own quadratic error mesh decimation here
    mesh_decimated = simplify_quadric_error(mesh, face_count=4)
    
    # # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('../writeup/cube_decimated3.obj')