import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from numpy.linalg import inv
import math

def normalize_coordinates(matriz):
    min_val = np.min(matriz)
    max_val = np.max(matriz)
    normalized_matrix = (matriz - min_val) / (max_val - min_val)
    return normalized_matrix

def printInterval(title, control_array_x, control_array_y, array_x, array_y):
    plt.scatter(array_x, array_y, color = "dodgerblue")
    plt.scatter(control_array_x, control_array_y, color = "red")
    plt.title(title)
    plt.show()

def rasterizar_linha(coord, num_fragments=100):
    # Calcular as diferenças entre coordenadas
    dx = coord[1, 0] - coord[0, 0]
    dy = coord[1, 1] - coord[0, 1]

    # Criar vetor de incrementos
    increments = np.array([dx / num_fragments, dy / num_fragments])

    # Calcular todos os fragmentos
    fragmentos = [coord[0] + i * increments for i in range(num_fragments + 1)]

    return np.concatenate(fragmentos)

def hermite_blend(P_points, T_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    T = np.matrix([t**3, t**2, t, np.ones(t.shape[0])])
    G = np.vstack((P_points, T_points))
    H = np.matrix([[2,-2,1,1], [-3,3,-2,-1], [0,0,1,0], [1,0,0,0]])
    P = T.T * H * G
    results_divided = [rasterizar_linha(np.vstack((P[i], P[i+1]))) for i in range(0, len(P), 2)]
    return np.concatenate(results_divided)

def calculate_blend_for_segments(points, tangents):
    segments = []
    for i in range(len(points) - 1):
        start_index = i
        end_index = i + 2
        blended_segment = hermite_blend(points[start_index:end_index], tangents[start_index:end_index])
        segments.append(blended_segment)
    return np.concatenate(segments, axis=0)

def find_intersections(x_or_y, constant, vertices, axis):
    intersections = []
    num_vertices = len(vertices)
    for i in range(num_vertices):
        p1 = vertices[i].A1.tolist()
        p2 = vertices[(i + 1) % num_vertices].A1.tolist()
        if (p1[axis] <= x_or_y < p2[axis]) or (p2[axis] <= x_or_y < p1[axis]):
            if p1[axis] != p2[axis]:
                other_axis = 1 - axis
                val = p1[other_axis] + (x_or_y - p1[axis]) * (p2[other_axis] - p1[other_axis]) / (p2[axis] - p1[axis])
                intersections.append(val)
    intersections.sort()
    return intersections

def scanline(polygons, step=1e-3):
    min_x, max_x = np.min(polygons[:, 0]), np.max(polygons[:, 0])
    min_y, max_y = np.min(polygons[:, 1]), np.max(polygons[:, 1])

    contour_points_set = set()
    fill_points = []

    for x in np.arange(min_x, max_x, step):
        intersections = find_intersections(x, min_y, polygons, axis=0)

        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                fill_points.append(rasterizar_linha(np.matrix([(x, intersections[i]), (x, intersections[i + 1])])))

        if intersections:
            contour_points_set.add((x, intersections[0]))
            contour_points_set.add((x, intersections[-1]))

    for y in np.arange(min_y, max_y, step):
        intersections = find_intersections(y, min_x, polygons, axis=1)

        if intersections:
            contour_points_set.add((intersections[0], y))
            contour_points_set.add((intersections[-1], y))

    contour_points = np.matrix(list(contour_points_set))

    fill_points = np.concatenate(fill_points)

    return contour_points, fill_points


INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000

def compute_out_code(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min:
        code |= LEFT
    elif x > x_max:
        code |= RIGHT
    if y < y_min:
        code |= BOTTOM
    elif y > y_max:
        code |= TOP
    return code

def cohen_sutherland_clip(line, x_min, y_min, x_max, y_max):
    x1, y1 = line[0, 0], line[0, 1]
    x2, y2 = line[1, 0], line[1, 1]

    out_code1 = compute_out_code(x1, y1, x_min, y_min, x_max, y_max)
    out_code2 = compute_out_code(x2, y2, x_min, y_min, x_max, y_max)
    accept = False

    while True:
        if out_code1 == 0 and out_code2 == 0:
            accept = True
            break
        elif (out_code1 & out_code2) != 0:
            break
        else:
            x = y = 0.0
            out_code_out = out_code1 if out_code1 != 0 else out_code2

            if out_code_out & TOP:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif out_code_out & BOTTOM:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif out_code_out & RIGHT:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif out_code_out & LEFT:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            if out_code_out == out_code1:
                x1, y1 = x, y
                out_code1 = compute_out_code(x1, y1, x_min, y_min, x_max, y_max)
            else:
                x2, y2 = x, y
                out_code2 = compute_out_code(x2, y2, x_min, y_min, x_max, y_max)

    if accept:
        return np.matrix([[x1, y1], [x2, y2]])
    else:
        return np.matrix([])

def clip_polygon(poly, x_min, y_min, x_max, y_max):
    def inside(p, bound, coord_index, limit):
        if bound == 'left':
            return p[coord_index] >= limit
        if bound == 'right':
            return p[coord_index] <= limit
        if bound == 'bottom':
            return p[coord_index] >= limit
        if bound == 'top':
            return p[coord_index] <= limit

    def compute_intersection(p1, p2, bound, coord_index, limit):
        if coord_index == 0:
            # x-coordinate intersection
            y = p1[1] + (p2[1] - p1[1]) * (limit - p1[0]) / (p2[0] - p1[0])
            return np.array([limit, y])
        else:
            # y-coordinate intersection
            x = p1[0] + (p2[0] - p1[0]) * (limit - p1[1]) / (p2[1] - p1[1])
            return np.array([x, limit])

    def clip_edge(poly, bound, coord_index, limit):
        new_poly = []
        p1 = poly[-1]
        for p2 in poly:
            if inside(p2, bound, coord_index, limit):
                if not inside(p1, bound, coord_index, limit):
                    new_poly.append(compute_intersection(p1, p2, bound, coord_index, limit))
                new_poly.append(p2)
            elif inside(p1, bound, coord_index, limit):
                new_poly.append(compute_intersection(p1, p2, bound, coord_index, limit))
            p1 = p2
        return new_poly

    polygon = poly.tolist()
    polygon = clip_edge(polygon, 'left', 0, x_min)
    polygon = clip_edge(polygon, 'right', 0, x_max)
    polygon = clip_edge(polygon, 'bottom', 1, y_min)
    polygon = clip_edge(polygon, 'top', 1, y_max)

    if len(polygon) > 0:
        return np.matrix(polygon)
    else:
        return np.matrix([])

def plot_polygons_points(polygon, clipped_polygon, x_min, y_min, x_max, y_max):
    fig, ax = plt.subplots()
    ax.scatter(polygon[:, 0].A1, polygon[:, 1].A1, color='blue', label='Original Polygon Points')

    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=None, edgecolor='r', linewidth=2, label='Clipping Window')
    ax.add_patch(rect)

    # Verificar se existe um polígono recortado para plotar
    if clipped_polygon.size > 0:
        ax.scatter(clipped_polygon[:, 0].A1, clipped_polygon[:, 1].A1, color='green', label='Clipped Polygon Points')

    # Configurações adicionais do gráfico
    ax.set_title('Polygon Clipping Points')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()