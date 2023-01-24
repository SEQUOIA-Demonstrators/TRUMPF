import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPoint
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
from QUBO_SA_SOLVER import *
from TSP_BATCH_SOLVER_QAOA import *
from RECTANGLE_PACKER import RECTANGLE_PACKER
import pickle
import random
import copy
import time
from datetime import datetime


class QUANTUM_PACKER():

    def __init__(self, instance_name, container_height, pieces, n_rotations, tsp_solver, distance_threshold, max_cluster_size,
                 n_partitions, num_qubits,
                 backend, solver_backend):
        """

        :param instance_name:
        :param container_height: fixed height of the container
        :param pieces:
        :param n_rotations:
        :param tsp_solver: string, one of:
                           BF (brute force),
                           SA (simulated annealing) or
                           QAOA (Quantum Approximate Optimization Algorithm) or
                           QAOA+ (Quantum Alternating Operator Ansatz)
        :param distance_threshold: maximum geometric incompatibility between pieces for clustering
        :param max_cluster_size:
        :param n_partitions: number of partitions to filter
        :param num_qubits:
        :param backend:
        :param solver_backend:
        """
        self.instance_name = instance_name
        self.H = container_height
        self.pieces = [Polygon(piece) for piece in pieces]
        self.num_pieces = len(self.pieces)
        self.n_rotations = n_rotations
        self.TSP_solver = tsp_solver
        self.distance_threshold = distance_threshold
        self.num_qubits = num_qubits
        self.max_cluster_size = max_cluster_size
        self.n_partitions = n_partitions
        self.alpha = 10.0 # percentage increase for length relaxation
        self.beta = 10.0 # percentage increase for height relaxation
        self.num_relaxations = 3
        self.phi_set = np.arange(0, 360, 5)
        self.theta_set = np.arange(0, 360, 360.0 / n_rotations)
        self.radius_step = 5
        # QAOA setting
        if tsp_solver == 'QAOA':
            self.use_approximate_optimization = True
            self.optimizer = 'COBYLA'
            self.num_repeats = 5
        elif tsp_solver == 'QAOA+':
            self.use_approximate_optimization = False
            self.optimizer = 'COBYLA'
            self.num_repeats = 4
        self.num_shots = 1000
        # Quantum backend
        self.backend = backend
        self.solver_backend = solver_backend

    def center_of_shape(self, shape):
        """
        Centroid of the shape.
        :param shape: Shapely geometry
        :return: a Point
        """
        center = shape.centroid
        return Point(center.x, center.y)

    def rotate_shape(self, shape, rotation_angle, rotation_center=None):
        """
        Rotates a shape around a center.
        :param shape: a Shapely geometry
        :param rotation_angle: angle in degrees
        :param rotation_center: center of rotation, by default set to the center of the shape.
        :return:
        """
        if str(type(shape)) == '<class \'list\'>':
            shape = MultiPolygon(shape)
        if rotation_center == None:
            rotation_center = self.center_of_shape(shape)
        try:
            rotated_shape = affinity.rotate(shape, rotation_angle, rotation_center)
        except:
            pass
        return rotated_shape

    def translate_shape(self, shape, vector):
        """
        Translates a shape.
        :param shape: Shapely geometry
        :param vector: translation vector as a Point
        :return: translated shape
        """
        translated_shape = affinity.translate(shape, xoff=vector.x, yoff=vector.y)
        return translated_shape

    def place_shape(self, shape, position, angle):
        """
        Rotates a shape and places its center at the desired position.
        :param shape: Shapely geometry
        :param position: Point where to place the center
        :param angle: rotation angle in degrees
        :return: placed shape
        """
        placed_shape = self.rotate_shape(shape, angle)
        c = self.center_of_shape(placed_shape)
        translation_vector = Point(position.x - c.x, position.y - c.y)
        placed_shape = self.translate_shape(placed_shape, translation_vector)
        return placed_shape

    def is_intersection(self, shape1, shape2):
        """
        Test if two shapes intercept.
        :param shape1: Shapely geometry
        :param shape2: Shapely geometry
        :return: True of False
        """
        # intersection of a polygon and a polygon
        if (str(type(shape1)) == '<class \'shapely.geometry.polygon.Polygon\'>') and (str(type(shape2)) == '<class \'shapely.geometry.polygon.Polygon\'>'):
            if shape1.intersection(shape2).is_empty:
                return False
            else:
                return True
        # intersection of a segment [A,B] with a polygon
        elif str(type(shape1)) == '<class \'shapely.geometry.linestring.LineString\'>' and (str(type(shape2)) == '<class \'shapely.geometry.polygon.Polygon\'>'):
            A = shape1.boundary[0]
            B = shape1.boundary[1]
            L = shape1.length
            u_x = B.x - A.x
            u_y = B.y - A.y
            # for points P in the open segment (A,B)
            for i in range(1, 10):
                P = Point([A.x + i * u_x / 10.0, A.y + i * u_y / 10.0])
                # if P is in shape2
                if shape2.contains(P):
                    # the segment intersects shape2
                    return True
            # otherwise there is no intersection
            return False
        # intersection of a list of polygons and a polygon
        elif str(type(shape1)) == '<class \'list\'>' and (str(type(shape2)) == '<class \'shapely.geometry.polygon.Polygon\'>'):
            for polygon in shape1:
                if self.is_intersection(polygon, shape2):
                    return True
            return False
        # intersection of a polygon and a list of polygons
        elif str(type(shape1) == '<class \'shapely.geometry.polygon.Polygon\'>') and str(type(shape2)) == '<class \'list\'>':
            for polygon in shape2:
                if self.is_intersection(shape1, polygon):
                    return True
            return False
        # intersection of a multipolygon and a polygon
        elif str(type(shape1)) == '<class \'shapely.geometry.multipolygon.MultiPolygon\'>' and (str(type(shape2)) == '<class \'shapely.geometry.polygon.Polygon\'>'):
            return self.is_intersection(list(shape1), shape2)
        else:
            print('WARNING: UNCOVERED SITUATION IN QUANTUM_PACKER.is_intersection')

    def open_segment(self, A, B):
        """
        Creates the open segment between two points.
        :param A: Point
        :param B: Point
        :return: LineString
        """
        u_AB = np.array([B.x - A.x, B.y - A.x])
        u_AB = u_AB / np.linalg.norm(u_AB, 2)
        A_prime = Point(A.x + u_AB[0], A.y + u_AB[1])
        B_prime = Point(B.x - u_AB[0], B.y - u_AB[1])
        return LineString([A_prime, B_prime])

    def get_vertices(self, piece):
        """
        Vertices of a Polygon.
        :param piece: Shapely geometry
        :return: list of Points
        """
        if str(type(piece)) == '<class \'shapely.geometry.polygon.Polygon\'>':
            vertices = list(piece.exterior.coords)
            vertices = [Point(e) for e in vertices]
        else:
            vertices = []
            for polygon in piece.geoms:
                vertices += self.get_vertices(polygon)
        return vertices

    def distance(self, e1, e2):
        """
        Minimum Euclidean distance between two geometries.
        :param e1: geometry
        :param e2: geometry
        :return: scalar
        """
        return e1.distance(e2)

    @staticmethod
    def get_bounding_polygon(geometry):
        if type(geometry) == list:
            geometry = MultiPolygon(geometry)
        bounding_polygon = geometry.convex_hull
        area = bounding_polygon.area
        return bounding_polygon, area

    def is_facing(self, v1, v2, piece1, piece2):
        v1_v2 = self.open_segment(v1, v2)
        cond1 = self.is_intersection(v1_v2, piece1)
        cond2 = self.is_intersection(v1_v2, piece2)
        if (cond1 is False) and (cond2 is False):
            return True
        else:
            return False

    def facing_vertices(self, piece1, piece2, show_result=False):
        # set of vertices of piece1
        V1 = self.get_vertices(piece1)
        # set of vertices of piece2
        V2 = self.get_vertices(piece2)
        # facing vertices
        F1 = []
        F2 = []
        for v1 in V1:
            for v2 in V2:
                if self.is_facing(v1, v2, piece1, piece2):
                    F1.append(v1)
                    break
        for v2 in V2:
            for v1 in V1:
                if self.is_facing(v2, v1, piece2, piece1):
                    F2.append(v2)
                    break
        if show_result:
            fig, axs = plt.subplots()
            axs.set_aspect('equal', 'datalim')
            xs, ys = piece1.exterior.xy
            axs.fill(xs, ys, alpha=1, fc='white', ec='black')
            xs, ys = piece2.exterior.xy
            axs.fill(xs, ys, alpha=1, fc='white', ec='black')
            for vertex in V1 + V2:
                axs.scatter(vertex.x, vertex.y, c='grey')
            for vertex in F1:
                axs.scatter(vertex.x, vertex.y, c='black')
            for vertex in F2:
                axs.scatter(vertex.x, vertex.y, c='black')
            plt.title('Facing vertices')
            plt.show()
        return F1, F2

    def distance_between_pieces(self, piece1, piece2):
        """
        Computes the distance between two pieces that do not overlap, defined as the complement area inside their
        convex hull.

        :param piece1: Shapely geometry
        :param piece2: Shapely geometry
        :return: wasted area and convex hull
        """
        area_of_pieces = piece1.area + piece2.area
        convex_hull = MultiPolygon([piece1, piece2]).convex_hull
        return convex_hull.area - area_of_pieces, convex_hull

    def compute_radius(self, shape):
        """
        Radius of a shape, defined as the maximum distance of a point of the shape to its center.
        :param shape: Polygon
        :return: radius
        """
        c = self.center_of_shape(shape)
        E = self.get_vertices(shape)
        radius = 0
        for e in E:
            d = e.distance(c)
            if d > radius:
                radius = d
        return radius

    def place_shape_relative_to(self, shape2, shape1, translation_vector, orientation_shape2):
        # rotate shape2 around its center
        placed_shape_2 = self.rotate_shape(shape2, orientation_shape2)
        # reposition center of shape2 on the center of shape1
        center1 = self.center_of_shape(shape1)
        center2 = self.center_of_shape(shape2)
        placed_shape_2 = self.translate_shape(placed_shape_2, Point([center1.x - center2.x, center1.y - center2.y]))
        # translate the center of shape2 wrt to the center of shape1
        placed_shape_2 = self.translate_shape(placed_shape_2, translation_vector)
        return placed_shape_2

    def compute_minimum_distance_between_shapes_at_angle(self, shape1, shape2, phi, show_result=False):
        d_min = np.inf
        gi_star = 0
        best_orientation_2 = 0
        u_phi = [math.cos(phi * math.pi / 180), math.sin(phi * math.pi / 180)]
        max_radius = self.compute_radius(shape1) + self.compute_radius(shape2)
        for orientation_shape2 in self.theta_set:
            radius = 0
            while radius <= max_radius + self.radius_step:
                translation_vector = Point(radius * u_phi[0], radius * u_phi[1])
                placed_shape2 = self.place_shape_relative_to(shape2, shape1, translation_vector, orientation_shape2)
                if self.is_intersection(shape1, placed_shape2) == False:
                    d, ch = self.distance_between_pieces(shape1, placed_shape2)
                    if d < d_min:
                        d_min = d
                        best_orientation_2 = orientation_shape2
                        best_translation = translation_vector
                        best_placed_shape2 = placed_shape2
                        gi_star = d_min / ch.area
                radius += self.radius_step
        if show_result:
            if str(type(shape1)) == '<class \'shapely.geometry.polygon.Polygon\'>':
                list_of_polygons = [shape1, best_placed_shape2]
            else:
                list_of_polygons = [polygon for polygon in shape1.geoms] + [best_placed_shape2]
            title = 'for phi=' + "{:.1f}".format(phi) + ': ' + 'd=' +"{:.2f}".format(d_min) + \
                    ' using theta=' + "{:.1f}".format(best_orientation_2) + ' and tau=[' + \
                    "{:.1f}".format(best_translation.x) + ',' + "{:.1f}".format(best_translation.y) + ']'
            self.plot_Multipolygon(MultiPolygon(list_of_polygons), title)
        return d_min, best_orientation_2, best_translation, best_placed_shape2, gi_star

    def minimum_distance_between_shapes_at_angle(self, shape1, shape2, phi, shape1_index, shape2_index, show_result=False):
        # file where to store the these calculations
        storage_file = 'nofit_function_memory_' + self.instance_name + '.p'
        # check if the storage file exists
        try:
            # load the dictionary
            saved_calculations_dict = pickle.load(open(storage_file, 'rb'))
        except:
            # create an empty dictionary
            saved_calculations_dict = {}
            # save the file
            pickle.dump(saved_calculations_dict, open(storage_file, 'wb'))
        # key associated to the calculation needed
        key_calculation = str(shape1_index) + '; ' + str(shape2_index) + '; ' + str(phi)
        # if the calculation has already been made
        if key_calculation in saved_calculations_dict.keys():
            # get the results of the calculation
            (d_min, best_orientation_2, best_translation, best_placed_shape2, gi_star) = saved_calculations_dict[key_calculation]
        # otherwise
        else:
            # execute the calculation
            d_min, best_orientation_2, best_translation, best_placed_shape2, gi_star = self.compute_minimum_distance_between_shapes_at_angle(shape1, shape2, phi)
            # store NFF_{shape1,shape2}(phi) = (d_min, best_orientation_2, best_translation, best_placed_shape2)
            saved_calculations_dict.update({key_calculation: (d_min, best_orientation_2, best_translation, best_placed_shape2, gi_star)})
            # store also the implied value for NFF_{shape2,shape1}(phi)
            phi_prime = (phi+180-best_orientation_2) % 360
            key_calculation_converse = str(shape2_index) + '; ' + str(shape1_index) + '; ' + str(phi_prime)
            best_orientation_1_converse = - best_orientation_2 + 360
            r = np.sqrt(best_translation.x ** 2 + best_translation.y ** 2)
            best_translation_converse = Point([r * math.cos(phi_prime * math.pi / 180), r * math.sin(phi_prime * math.pi / 180)])
            best_placed_shape1 = self.place_shape_relative_to(shape1, shape2, best_translation_converse,
                                                              best_orientation_1_converse)
            saved_calculations_dict.update({key_calculation_converse: (d_min,
                                                                       best_orientation_1_converse,
                                                                       best_translation_converse, best_placed_shape1, gi_star)})
            # save the file
            pickle.dump(saved_calculations_dict, open(storage_file, 'wb'))
        # return the results
        return d_min, best_orientation_2, best_translation, best_placed_shape2, gi_star

    def minimum_distance_between_shapes(self, piece1, piece2, piece1_name, piece2_name):
        d_min_star = np.inf
        gi_star = 0
        best_placed_shape2_star = None
        # for each orbiting angle phi
        for phi in self.phi_set:
            # compute the value of the nofit function
            piece1_index = piece1_name[1:]
            piece2_index = piece2_name[1:]
            d_min, best_orientation_2, best_translation, best_placed_shape2, gi = self.minimum_distance_between_shapes_at_angle(piece1, piece2, phi, piece1_index, piece2_index)
            if d_min < d_min_star:
                d_min_star = d_min
                gi_star = gi
                best_placed_shape2_star = best_placed_shape2
        # create a new plot
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        # plot the convex hull area in grey
        _, ch = self.distance_between_pieces(piece1, best_placed_shape2_star)
        xs, ys = ch.exterior.xy
        axs.fill(xs, ys, alpha=1.0, fc='grey', ec='black')
        # plot piece1 in white
        xs, ys = piece1.exterior.xy
        axs.fill(xs, ys, alpha=1.0, fc='white', ec='black')
        # plot best_placed_shape2_star in white
        xs, ys = best_placed_shape2_star.exterior.xy
        axs.fill(xs, ys, alpha=1.0, fc='white', ec='black')
        # add title
        title = 'd(' + piece1_name + ', ' + piece2_name + ') = ' + "{:.1f}".format(d_min_star) + ' gi=' + "{:.3f}".format(gi_star)
        plt.title(title)
        # show plot
        plt.show()
        plt.show()
        # save plot
        file_name = 'd(' + piece1_name + ', ' + piece2_name + ').pdf'
        fig.savefig(file_name)
        # return the distance
        return d_min_star, gi_star

    def identical_pieces_lower_index_dict(self):
        # for each piece index, determines which list of pieces of strictly lower index are the same
        identical_pieces_dict = {}
        for i in range(self.num_pieces):
            pieces_identical_to_i = []
            for j in range(i):
                if self.pieces[i].equals(self.pieces[j]):
                    pieces_identical_to_i.append(j)
            identical_pieces_dict.update({i: pieces_identical_to_i})
        return identical_pieces_dict

    def compute_distance_matrix(self):
        # file where to store the distance matrix
        storage_file = 'distance_matrix_' + self.instance_name + '.p'
        storage_file_incompatibility = 'incompatibility_matrix_' + self.instance_name + '.p'
        # check if the distance matrix has already been computed and saved locally
        try:
            D = pickle.load(open(storage_file, 'rb'))
            GI = pickle.load(open(storage_file_incompatibility, 'rb'))
            print('The distance matrix and nofit functions have already been pre-computed and will be directly loaded')
        except:
            print('The distance matrix and nofit functions need to be pre-computed. Please wait...')
            num_shapes = len(self.pieces)
            # for each piece index, determines which list of pieces of strictly lower index are the same
            identical_pieces_dict = self.identical_pieces_lower_index_dict()
            # compute distance matrix avoiding recomputing distances between identical pieces
            D = np.zeros((num_shapes, num_shapes))
            GI = np.zeros((num_shapes, num_shapes))
            computed_distances = []
            for i in range(num_shapes):
                pieces_identical_to_i = identical_pieces_dict[i]
                if len(pieces_identical_to_i) == 0:
                    for j in range(num_shapes):
                        pieces_identical_to_j = identical_pieces_dict[j]
                        if len(pieces_identical_to_j) == 0:
                            d_ij, gi_ij = self.minimum_distance_between_shapes(self.pieces[i], self.pieces[j], 'P'+str(i), 'P'+str(j))
                            distance_key = str(i) + '-' + str(j)
                            computed_distances.append(distance_key)
                        else:
                            first_piece_identical_to_j = min(pieces_identical_to_j)
                            if (first_piece_identical_to_j < j):
                                distance_key = str(i) + '-' + str(first_piece_identical_to_j)
                                if distance_key in computed_distances:
                                    d_ij = D[i, first_piece_identical_to_j]
                                    gi_ij = GI[i, first_piece_identical_to_j]
                                else:
                                    d_ij, gi_ij = self.minimum_distance_between_shapes(self.pieces[i], self.pieces[j])
                                    computed_distances.append(distance_key)
                            else:
                                d_ij, gi_ij = self.minimum_distance_between_shapes(self.pieces[i], self.pieces[j])
                                distance_key = str(i) + '-' + str(j)
                                computed_distances.append(distance_key)
                        # store d_ij in the distance matrix
                        D[i, j] = d_ij
                        GI[i, j] = gi_ij
                        print('Computed distance between P' + str(i) + ' and P' + str(j))
                else:
                    first_piece_identical_to_i = min(pieces_identical_to_i)
                    D[i, :] = D[first_piece_identical_to_i, :]
                    D[:, i] = D[i, :]
                    GI[i, :] = GI[first_piece_identical_to_i, :]
                    GI[:, i] = GI[i, :]
            # round D to first decimal
            D = np.round(D, decimals=1)
            GI = np.round(GI, decimals=3)
            # save the distance matrix
            pickle.dump(D, open(storage_file, 'wb'))
            pickle.dump(GI, open(storage_file_incompatibility, 'wb'))
            print('Computation done! Saved matrix in ' + storage_file)
        # print matrix
        print(D)
        # show matrix in latex code
        print('$$ D = \left[ \\begin{array}{rrrrrrrr}')
        for i in range(D.shape[0]):
            row = ''
            values = ["{:.1f}".format(D[i, j]) for j in range(D.shape[1])]
            row = ' & '.join(values) + ' \\\\'
            print(row)
        print('\\end{array}\\right] $$')

        print('$$ GI = \left[ \\begin{array}{rrrrrrrr}')
        for i in range(GI.shape[0]):
            row = ''
            values = ["{:.3f}".format(GI[i, j]) for j in range(GI.shape[1])]
            row = ' & '.join(values) + ' \\\\'
            print(row)
        print('\\end{array}\\right] $$')
        # return the distance matrix
        return D, GI

    def decode_TSP_solution(self, results_dictionary):
        num_cities = int(math.sqrt(len(results_dictionary.keys())))
        hamiltonian_path = np.zeros(num_cities)
        for var in results_dictionary.keys():
            parsed_variable_name = var.split('_')
            i = int(parsed_variable_name[1])
            p = int(parsed_variable_name[2])
            value = results_dictionary[var]
            if value == 1:
                hamiltonian_path[p-1] = i
        return hamiltonian_path

    @staticmethod
    def merge(p1, p2):
        """
        Merges two geometric figures into one.
        :param p1: Shapely geometry
        :param p2: Shapely geometry
        :return: Shapely geometry
        """
        if str(type(p1)) == '<class \'shapely.geometry.multipolygon.MultiPolygon\'>':
            list_of_polygons = list(p1) + [p2]
            result = MultiPolygon(list_of_polygons)
        else:
            if str(type(p1)) != '<class \'list\'>':
                result = MultiPolygon([p1, p2])
            else:
                result = MultiPolygon(p1 + [p2])
        return result

    def greedy_packer(self, pieces, show_result=False):
        """
        Packs a list of pieces one by one and in the order given by the list, trying at any step to minimize the area
        of the bounding box.

        :param pieces: list of pieces as Polygons
        :return: all pieces packed as a MultiPolygon
        """
        # number of pieces to pack
        num_pieces = len(pieces)
        # initialize the packed pieces with the first piece
        packed_pieces = [self.pieces[pieces[0]]]
        # position of center and orientation of the last piece packed
        tau_last = self.center_of_shape(self.pieces[pieces[0]])
        theta_last = 0
        previous_placed_piece_to_pack = self.pieces[pieces[0]]
        # for following pieces i to pack
        for i in range(1, num_pieces):
            # the lastly packed piece
            last_piece_packed = pieces[i-1]
            # the piece to pack now
            piece_to_pack = pieces[i]
            # initialize status of packing
            status_packing = 'fail'
            # initialize the wasted area of the bounding box with which the new pieces can be packed
            w_star = np.inf
            # for every possible rotation angle phi of piece_to_pack around last_piece_packed
            for phi in self.phi_set:
                # determine theta_2 and translation_2 using the nofit function
                _, theta_2, translation_2, _, _ = self.minimum_distance_between_shapes_at_angle(self.pieces[last_piece_packed], self.pieces[piece_to_pack], phi, last_piece_packed, piece_to_pack)
                # position the new piece to pack relative to the last one
                theta_new = theta_last + theta_2
                r = math.sqrt(translation_2.x ** 2 + translation_2.y ** 2)
                tau_new = Point([r * math.cos((theta_last + phi) * math.pi / 180), r * math.sin((theta_last + phi) * math.pi / 180)])
                placed_piece_to_pack = self.place_shape_relative_to(self.pieces[piece_to_pack], previous_placed_piece_to_pack, tau_new, theta_new)
                # compute the new waste
                _, new_w = self.fit_bounding_box(self.merge(packed_pieces, placed_piece_to_pack))
                # if the placement at angle phi given by the nofit function for the piece to pack is valid (avoids overlapping)
                if self.is_intersection(packed_pieces, placed_piece_to_pack) == False:
                    # if the placement at angle phi reduces the area of the bounding box
                    if new_w < w_star:
                        # record the best distance and the placement for the piece to pack
                        w_star = new_w
                        placed_piece_to_pack_star = placed_piece_to_pack
                        theta_star = theta_new
                        tau_star = tau_new
                        # update the status of packing
                        status_packing = 'success'
                        # show result
                        #if show_result:
                        #    title = ' improved for phi=' + "{:.1f}".format(phi) + ' with theta=' + "{:.1f}".format(theta_star)
                        #    self.plot_Multipolygon(self.merge(packed_pieces, placed_piece_to_pack), title)
                # else, the placement given by the nofit function at angle phi has failed
                else:
                    # keep theta_2 but keep translating the piece in the direction of translation_2 until the piece fits
                    translation_2_norm = math.sqrt(translation_2.x ** 2 + translation_2.y ** 2)
                    delta_translation_2 = Point([self.radius_step * translation_2.x / translation_2_norm, self.radius_step * translation_2.y / translation_2_norm])
                    while self.is_intersection(packed_pieces, placed_piece_to_pack) == True:
                        # enlarge a bit translation_2
                        translation_2 = Point([translation_2.x + delta_translation_2.x,
                                               translation_2.y + delta_translation_2.y])
                        # re-position the new piece to pack relative to the last one
                        theta_new = theta_last + theta_2
                        r = math.sqrt(translation_2.x ** 2 + translation_2.y ** 2)
                        tau_new = Point([tau_last.x + r * math.cos((theta_last + phi) * math.pi / 180),
                                         tau_last.y + r * math.sin((theta_last + phi) * math.pi / 180)])
                        placed_piece_to_pack = self.place_shape_relative_to(self.pieces[piece_to_pack], self.pieces[last_piece_packed], tau_new, theta_new)
                    # compute the new waste
                    _, new_w = self.fit_bounding_box(self.merge(packed_pieces, placed_piece_to_pack))
                    # if the placement at angle phi reduces the area of the bounding box
                    if new_w < w_star:
                        # record the best distance and the placement for the piece to pack
                        w_star = new_w
                        placed_piece_to_pack_star = placed_piece_to_pack
                        theta_star = theta_new
                        tau_star = tau_new
                        # update the status of packing
                        status_packing = 'success'
                        # show result
                        #if show_result:
                        #    title = 'improved for phi=' + "{:.1f}".format(phi) + ' with theta=' + "{:.1f}".format(theta_star)
                        #    self.plot_Multipolygon(self.merge(packed_pieces, placed_piece_to_pack), title)
            # if packing piece i was possible
            if status_packing == 'success':
                # greedily pack the piece to pack with the others according to the best placement found
                packed_pieces = copy.deepcopy(self.merge(packed_pieces, placed_piece_to_pack_star))
                # update position center and orientation of the last piece packed
                tau_last = tau_star
                theta_last = theta_star
                previous_placed_piece_to_pack = placed_piece_to_pack_star
            # otherwise, packing is impossible and must be aborted
            else:
                packed_pieces = None
                break
        if show_result:
            title = 'Greedy packing of pieces ' + '-'.join([str(p) for p in pieces])
            self.plot_Multipolygon(packed_pieces, title)
        return packed_pieces

    def plot_Multipolygon(self, multi_polygon, my_title=''):
        # if multi_polygon is just a polygon then make it a list of one polygon
        if str(type(multi_polygon)) == '<class \'shapely.geometry.polygon.Polygon\'>':
            multi_polygon = MultiPolygon([multi_polygon])
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        for geom in multi_polygon:
            xs, ys = geom.exterior.xy
            axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
        plt.title(my_title)
        plt.show()

    def show_pieces(self):
        num_pieces = len(self.pieces)
        num_pieces_per_row = int(math.sqrt(num_pieces))
        W = 0
        H = 0
        for i, p in enumerate(self.pieces):
            bbox = p.bounds
            minx = bbox[0]
            miny = bbox[1]
            maxx = bbox[2]
            maxy = bbox[3]
            width = maxx - minx
            height = maxy - miny
            if width > W:
                W = width
            if height > H:
                H = height
        cell_spacing = 20
        W += cell_spacing
        H += cell_spacing
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        row = 1
        column = 1
        for i, p in enumerate(self.pieces):
            position = Point([row * W, column * H])
            p_placed = self.place_shape(p, position, 0)
            xs, ys = p_placed.exterior.xy
            axs.fill(xs, ys, alpha=1.0, fc='white', ec='black')
            center = self.center_of_shape(p_placed)
            axs.plot((center.x), (center.y), 'o', color='k')
            axs.text(center.x + 30, center.y - 30, 'P'+str(i))
            if column >= num_pieces_per_row:
                column = 1
                row += 1
            else:
                column += 1
        plt.title(self.instance_name + ': ' + str(num_pieces) + ' pieces')
        plt.show()
        file_name = 'PIECES ' + self.instance_name + '.pdf'
        fig.savefig(file_name)

    def fit_bounding_box(self, multi_polygon, show_result=False):
        """
        Returns the general minimum bounding rectangle that contains the object.
        This rectangle is not constrained to be parallel to the coordinate axes.
        """
        # if multi_polygon is just a polygon then make it a list of one polygon
        if str(type(multi_polygon)) == '<class \'shapely.geometry.polygon.Polygon\'>':
            multi_polygon = MultiPolygon([multi_polygon])
        # if multi_polygon is a list of polygons, then make it a MultiPolygon
        if str(type(multi_polygon)) == '<class \'list\'>':
            multi_polygon = MultiPolygon(multi_polygon)
        # find rectangular bounding box with minimum area not parallel to the axis
        bounding_box = multi_polygon.minimum_rotated_rectangle
        # show result of fitting a bounding box around the pieces
        if show_result:
            fig, axs = plt.subplots()
            axs.set_aspect('equal', 'datalim')
            xs, ys = bounding_box.exterior.xy
            axs.fill(xs, ys, alpha=0.5, fc='b', ec='none')
            for geom in multi_polygon.geoms:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            plt.title('Bounding box')
            plt.show()
        bounding_box_area = bounding_box.area
        return bounding_box, bounding_box_area

    def get_bounding_box_dimensions(self, polygon):
        x_min = polygon.bounds[0]
        y_min = polygon.bounds[1]
        x_max = polygon.bounds[2]
        y_max = polygon.bounds[3]
        # get length of bounding box edges
        dimensions = (x_max - x_min, y_max - y_min)
        return dimensions

    @staticmethod
    def build_EXACT_COVER_objective_function(N, S):
        """
        Given a set of N elements, X={c_1, ..., c_N}, a family of M subsets of X,
        S={S_1, ..., S_M} such that S_i \subset X and \cup_{i=1}^M S_i = X, find a
        subset I of {1, ..., M} such that \cup_{i \in I} S_i = X where S_i \cap S_j = \emptyset
        for i \neq j \in I.

        :param N: number of elements in X
        :param S: list of lists of integers in [1, ..., N]
        :return: list I of indices of the sets selected from S
        """
        # number of subsets in S
        M = len(S)
        # build list of binary variables s_i, i \in {1, ..., M}
        binary_variables = []
        for i in range(1, M + 1):
            binary_variables.append('s_' + str(i))
        # build list of coefficients w_i of the binary variables s_i
        linear_coefficients = []
        for i in range(1, M + 1):
            # coefficient of s_i is w_i = - 2 * |S_i|
            w_i = - 2 * len(S[i-1])
            linear_coefficients.append(w_i)
        # build dictionary of quadratic coefficients
        quadratic_coefficients = {}
        for i in range(1, M + 1):
            for k in range(1, M + 1):
                # coefficient of s_i * s_k is |S_i \cap S_k|
                w_ik = len([e for e in S[i-1] if e in S[k-1]])
                s_i = 's_' + str(i)
                s_k = 's_' + str(k)
                quadratic_coefficients.update({(s_i, s_k): w_ik})
        return binary_variables, linear_coefficients, quadratic_coefficients

    def pack_rectangles(self, rectangles, rectangle_names, height_limit):
        my_rect_packer = RECTANGLE_PACKER(rectangles, rectangle_names, height_limit)
        rectangles_arrangement = my_rect_packer.solve()
        return rectangles_arrangement

    def unpack_rectangles(self, rectangles_arrangement, R):
        # initialize the layout (list of polygons)
        layout = []
        # initialize the rectangle layout (list of rectangles)
        rectangle_layout = []
        # plot the pieces
        for rectangle_id, value in rectangles_arrangement.items():
            # position of the rectangle in the board
            x, y, w, h, b = value[0], value[1], value[2], value[3], value[4]
            rectangle = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            # pieces packed in a bounding box
            packed_pieces = R[rectangle_id][0]
            bounding_box = R[rectangle_id][1]
            # determines the geometric transformation (translation and rotation) that maps the bounding box to the rectangle
            position_bounding_box = self.center_of_shape(bounding_box)
            position_rectangle = self.center_of_shape(rectangle)
            translation = Point(
                (position_rectangle.x - position_bounding_box.x, position_rectangle.y - position_bounding_box.y))
            superposition_ratio_star = 0.0
            for angle in range(360):
                placed_bounding_box = self.place_shape(bounding_box, position_rectangle, angle)
                superposition_ratio = rectangle.intersection(placed_bounding_box).area / rectangle.area
                if superposition_ratio > superposition_ratio_star:
                    superposition_ratio_star = superposition_ratio
                    angle_star = angle
                if superposition_ratio_star > 0.999:
                    break
            # move the packed pieces using this transformation
            packed_pieces = self.rotate_shape(packed_pieces, angle_star, position_bounding_box)
            packed_pieces = self.translate_shape(packed_pieces, translation)
            # if packed_pieces is just a polygon then make it a list of one polygon
            if str(type(packed_pieces)) == '<class \'shapely.geometry.polygon.Polygon\'>':
                packed_pieces = MultiPolygon([packed_pieces])
            # adds the packed pieces to the layout
            layout += list(packed_pieces)
            # adds the rectangle to the rectangle layout
            rectangle_layout.append(rectangle)
        return layout, rectangle_layout

    def is_in_upper_right_quadrant(self, piece):
        """
        Check that a piece is positionned within the quadrant x >= 0 and y >= 0
        :param translated_piece:
        :return: Boolean
        """
        V = self.get_vertices(piece)
        for v in V:
            if v.x < 0 or v.y < 0:
                return False
        return True

    def local_optimization(self, layout, container):
        """
        Moves all pieces of the layout to the left and down, while avoiding piece overlapping and going out of the container.

        :param layout: list of pieces
        :param container: a polygon
        :return: optimized layout
        """
        print('            running local optimization...')
        # prioritized list of unitary vectors for translating down and to the left
        prioritized_directions = []
        prioritized_directions.append(Point([-1, 0]))
        prioritized_directions.append(Point([-1, -1]))
        prioritized_directions.append(Point([-1, +1]))
        prioritized_directions.append(Point([0, -1]))
        is_optimal = False
        while is_optimal == False:
            is_optimal = True
            for i, piece in enumerate(layout):
                # list of pieces without i
                other_pieces = layout
                del other_pieces[i]
                # move the piece towards the origin as much as possible, while avoiding overlapping with the other ones
                keep_moving_piece = True
                while keep_moving_piece:
                    # test if we can move the piece in one of the allowed directions and if so do it
                    keep_moving_piece = False
                    # for each allowed direction
                    for vector in prioritized_directions:
                        # translate the piece
                        translated_piece = self.translate_shape(piece, vector)
                        # if the piece is still in the upper right quadrant (x >= 0 and y >= 0)
                        if self.is_in_upper_right_quadrant(translated_piece):
                            # check if the translated piece intersects another one
                            intersection_test = False
                            for other_piece in other_pieces:
                                if self.is_intersection(translated_piece, other_piece):
                                    intersection_test = True
                                    break
                            # and if the translated piece doesn't intersect another one
                            if not intersection_test:
                                # translate the piece
                                piece = translated_piece
                                # keep moving it
                                keep_moving_piece = True
                                # the layout is still not optimal
                                is_optimal = False
                                # don't look at the remaining directions
                                break
                # replace moved pieced i of the layout
                other_pieces[i:i] = [piece]
                layout = other_pieces
        # return optimal layout
        return layout

    def relocate_rightmost_piece(self, layout):
        """

        :param layout:
        :return:
        """
        # initialize the improved layout
        improved_layout = layout
        # finds the rightmost piece
        x_max = -np.inf
        rightmost_piece_index = -1
        rightmost_piece = None
        success = False
        for i, piece in enumerate(layout):
            piece_x_max = piece.bounds[2]
            if piece_x_max > x_max:
                x_max = piece_x_max
                rightmost_piece_index = i
                rightmost_piece = piece
        # delete the rightmost piece from the layout
        layout_without_rightmost_piece = copy.deepcopy(layout)
        del layout_without_rightmost_piece[rightmost_piece_index]
        # find a placement for the piece to relocate
        grid_step_x = x_max / 100
        grid_step_y = self.H / 100
        for x in np.arange(0, x_max, grid_step_x):
            for y in np.arange(0, self.H, grid_step_y):
                for theta in self.theta_set:
                    # place the piece at (x,y) with angle theta
                    repositionned_piece = self.place_shape(rightmost_piece, Point(x, y), theta)
                    # if the piece is in the upper right quadrant
                    if self.is_in_upper_right_quadrant(repositionned_piece):
                        # if the container height is not exceeded by the re-positioned piece
                        h = repositionned_piece.bounds[3]
                        if h <= self.H:
                            # check if the translated piece intersects another one
                            intersection_test = False
                            for other_piece in layout_without_rightmost_piece:
                                if self.is_intersection(repositionned_piece, other_piece):
                                    intersection_test = True
                                    break
                            # if the translated piece doesn't intersect another one
                            if not intersection_test:
                                # insert the re-positionned piece in the layout
                                new_layout = copy.deepcopy(layout_without_rightmost_piece)
                                new_layout.append(repositionned_piece)
                                # if the repositionned piece is more to the left that before
                                x_max_new = repositionned_piece.bounds[2]
                                if x_max_new < x_max:
                                    print('Shifting rightmost piece from x_max=' + str(x_max) + ' to ' + str(x_max_new))
                                    improved_layout = new_layout
                                    success = True
                                    return improved_layout, rightmost_piece_index, success
        print('Layout could not be improved by shifting the rightmost piece !')
        return improved_layout, rightmost_piece_index, success




        # return improved layout
        return improved_layout

    def required_height(self, multipolygon):
        multipolygon = MultiPolygon(multipolygon)
        # bounds of the multipolygon
        bounds = list(multipolygon.bounds)
        multipolygon_y_min = round(bounds[1], 3)
        multipolygon_y_max = round(bounds[3], 3)
        height = multipolygon_y_max - multipolygon_y_min
        return height

    def is_in_container(self, multipolygon, container):
        # bounds of the multipolygon
        bounds = list(multipolygon.bounds)
        multipolygon_x_min = round(bounds[0], 3)
        multipolygon_y_min = round(bounds[1], 3)
        multipolygon_x_max = round(bounds[2], 3)
        multipolygon_y_max = round(bounds[3], 3)
        # bounds of the container
        bounds = list(container.bounds)
        x_min = round(bounds[0], 3)
        y_min = round(bounds[1], 3)
        x_max = round(bounds[2], 3)
        y_max = round(bounds[3], 3)
        # test if the multipolygon is in the container
        if (multipolygon_x_min >= x_min) and (multipolygon_y_min >= y_min) and (multipolygon_x_max <= x_max) and (multipolygon_y_max <= y_max):
            return True
        else:
            return False

    def generate_partitions(self, num_partitions, min_cardinality, max_cardinality):
        """
        Generates a list of random partitions of the set {0, 1, ..., num_pieces-1}.

        :param num_partitions: number of partitions generates
        :param max_cardinality: maximum cardinality of elements in the partitions
        :return: list of partitions, each partition itself being a list of strings of the form '0-2-3-4-5-9'
        """
        partitions = []
        support = []
        for i in range(num_partitions):
            # initialize a new partition
            partition = []
            # initialize list of piece indices
            Q = list(range(self.num_pieces))
            while len(Q) > 0:
                # generate a random integer k between 1 and max_cardinality
                k = random.randint(min_cardinality, max_cardinality)
                # select randomly k elements from Q without replacement
                try:
                    P = random.sample(Q, k)
                # unless Q has size less than k
                except:
                    # take all pieces left in Q
                    P = Q
                # sort the indices in P
                P.sort()
                # add the P as element to the partition
                partition.append(P)
                # add P to the support, if P is not already in the support
                if P not in support:
                    support.append(P)
                # remove P from Q
                Q = [e for e in Q if e not in P]
            # add the partition to the list
            partitions.append(partition)
        # return the list of partitions
        return partitions, support

    def modulo_key(self, key):
        """
        Takes a key for a set of pieces, e.g. '0-4-12-15' and  replaces each piece index
        by the minimum index of an identical piece.

        :param key:
        :return: string
        """
        indices = key.split('-')
        indices = [int(s) for s in indices]
        minimum_indices = []
        identical_pieces_dict = self.identical_pieces_lower_index_dict()
        for index in indices:
            try:
                minimum_index = min(identical_pieces_dict[index])
            except:
                minimum_index = index
            minimum_indices.append(str(minimum_index))
        modulo_key = '-'.join(minimum_indices)
        return modulo_key

    def is_valid_path(self, result):
        num_vertices = int(math.sqrt(len(list(result.keys()))))
        x_matrix = np.zeros((num_vertices, num_vertices))
        for binary_variable in result.keys():
            strings = binary_variable.split('_')
            i = int(strings[1])
            p = int(strings[2])
            x_matrix[i-1, p-1] = result[binary_variable]
        for p in range(num_vertices):
            if np.sum(x_matrix[:, p]) != 1.0:
                return False
        for i in range(num_vertices):
            if np.sum(x_matrix[i, :]) != 1.0:
                return False
        return True

    def required_bin_length(self, layout):
        """
        Maximum x coordinate of a polygon in the layout
        :param layout: list of polygons
        :return: maximum x coordinate
        """
        return max([max(polygon.exterior.xy[0]) for polygon in layout])

    def build_TSP_objective_function(self, W):
        N = W.shape[0]
        # maximum of absolute distances
        Wmax = np.abs(W).max()
        # Penalty coefficient
        A = math.ceil(Wmax) + 1.0
        # build list of binary variables
        binary_variables = []
        for i in range(1, N+1):
            for p in range(1, N+1):
                binary_variables.append('x_'+str(i)+'_'+str(p))
        # build list of coefficients of the binary variables
        linear_coefficients = []
        for i in range(1, N+1):
            for p in range(1, N+1):
                # coefficient of x_i_p is -4A
                linear_coefficients.append(-4.0 * A)
        # build dictionary of quadratic coefficients
        quadratic_coefficients = {}

        for i in range(1, N+1):
            for j in range(1, N+1):
                for p in range(1, N):
                    var1 = 'x_'+str(i)+'_'+str(p)
                    var2 = 'x_'+str(j)+'_'+str(p+1)
                    quadratic_coefficients.update({(var1, var2): W[i-1, j-1]})

        for p in range(1, N+1):
            for i in range(1, N+1):
                for i_prime in range(1, N+1):
                    var1 = 'x_' + str(i) + '_' + str(p)
                    var2 = 'x_' + str(i_prime) + '_' + str(p)
                    quadratic_coefficients.update({(var1, var2): A})

        for i in range(1, N+1):
            for p in range(1, N+1):
                for p_prime in range(1, N+1):
                    var1 = 'x_' + str(i) + '_' + str(p)
                    var2 = 'x_' + str(i) + '_' + str(p_prime)
                    if (var1, var2) in quadratic_coefficients:
                        quadratic_coefficients.update({(var1, var2): A + quadratic_coefficients[(var1, var2)]})
                    else:
                        quadratic_coefficients.update({(var1, var2): A})

        return binary_variables, linear_coefficients, quadratic_coefficients

    @staticmethod
    def correct_path(result):
        # number of qubits
        num_qubits = int(np.sqrt(len(result)))
        # transform the result dictionary into an array
        X = np.zeros((num_qubits, num_qubits))
        for var in result.keys():
            parsed_variable_name = var.split('_')
            i = int(parsed_variable_name[1]) - 1
            p = int(parsed_variable_name[2]) - 1
            value = result[var]
            X[i, p] = value
        # make sure that the sums of x_{i,p} over p do not exceed 1
        for i in range(num_qubits):
            if np.sum(X[i, :]) > 1:
                active_columns = [p for p in range(num_qubits) if X[i, p] == 1]
                chosen_column = random.choice(active_columns)
                X[i, :] = np.zeros(num_qubits)
                X[i, chosen_column] = 1
        # make sure that the sums of x_{i,p} over i do not exceed 1
        for p in range(num_qubits):
            if np.sum(X[:, p]) > 1:
                active_rows = [i for i in range(num_qubits) if X[i, p] == 1]
                chosen_row = random.choice(active_rows)
                X[:, p] = np.zeros(num_qubits)
                X[chosen_row, p] = 1
        # make sure the sum of each row is not zero
        # for each row i
        for i in range(num_qubits):
            # if the row is empty
            if np.sum(X[i, :])==0:
                free_columns = [p for p in range(num_qubits) if np.sum(X[:, p]) == 0]
                chosen_column = random.choice(free_columns)
                X[i, chosen_column] = 1
        # make sure the sum of each column is not zero
        # for each column p
        for p in range(num_qubits):
            # if the column is empty
            if np.sum(X[:, p]) == 0:
                free_rows = [i for i in range(num_qubits) if np.sum(X[i, :]) == 0]
                chosen_row = random.choice(free_rows)
                X[chosen_row, p] = 1
        # convert the X array back to a dictionary
        result = {}
        for i in range(num_qubits):
            for p in range(num_qubits):
                var = 'x_' + str(i+1) + '_' + str(p+1)
                result.update({var: X[i, p]})
        # return corrected dictionary that now encodes a Hamiltonian path
        return result

    def cluster_distance(self, c1, c2, D):
        """
        Distance between two clusters.
        :param c1: list of indices of the elements in cluster 1
        :param c2: list of indices of the elements in cluster 1
        :param D: distance matrix between all elements as numpy array
        :return: minimum distance between elements of cluster 1 and 2
        """
        d_min = np.inf
        for i in c1:
            for j in c2:
                if D[int(i), int(j)] < d_min:
                    d_min = D[int(i), int(j)]
        return d_min

    def agglomerative_clustering(self, D, distance_threshold, max_cluster_size):
        """
        Single-linakge clustering algorithm.

        :param D: distance matrix
        :param distance_threshold: maxmium distance allowed for linking to clusters
        :param max_cluster_size: maximum number of elements per cluster
        :return: partition of the elements (list of lists)
        """
        # associate each piece to a cluster
        clusters = [[i] for i in range(self.num_pieces)]
        # as long as we can agglomerate clusters
        clusters_changed = True
        while clusters_changed==True:
            clusters_changed = False
            # for each cluster c1
            for c1 in range(len(clusters)-1):
                cluster1 = clusters[c1]
                # for each cluster c2 following c1
                for c2 in range(c1+1, len(clusters)):
                    cluster2 = clusters[c2]
                    # check if there are 2 pieces within the distance_threshold
                    if self.cluster_distance(cluster1, cluster2, D) <= distance_threshold:
                        # merge the 2 clusters
                        cluster12 = cluster1 + cluster2
                        cluster12.sort()
                        # if the merged cluster does not exceed the max size
                        if len(cluster12) <= max_cluster_size:
                            # replace clusters c1 and c2 by the merged cluster
                            if c1 <= c2:
                                clusters[c1] = cluster12
                                del clusters[c2]
                            else:
                                clusters[c2] = cluster12
                                del clusters[c1]
                            # flag change
                            clusters_changed = True
                            # break for loop c2
                            break
                if clusters_changed==True:
                    # break for loop c1
                    break
        # return the aggregated clusters
        return clusters

    def stochastic_agglomerative_clustering(self, D, distance_threshold, max_cluster_size, max_num_trials=1000):
        """
        Stochastic version of the single-linakge clustering algorithm.

        :param D: distance matrix
        :param distance_threshold: maxmium distance allowed for linking to clusters
        :param max_cluster_size: maximum number of elements per cluster
        :param max_num_trials: maximum number of trials to generate a random partition, default = 500
        :return: partition of the elements (list of lists)
        """

        def make_name_from_partition(_partition):
            elements_names = []
            for e in _partition:
                e_str = [str(x) for x in e]
                element_name = '-'.join(e_str)
                elements_names.append(element_name)
            name = '_'.join(elements_names)
            return name

        def single_linkage_distance(_c1, _c2):
            d_min = np.inf
            for p1 in _c1:
                for p2 in _c2:
                    if D[p1, p2] < d_min:
                        d_min = D[p1, p2]
            return d_min

        # start with one partition made of singletons
        partitions_dict = {}
        partition = [[i] for i in range(self.num_pieces)]
        partitions_dict.update({make_name_from_partition(partition): partition})

        num_trials = 0
        max_num_trials = 500
        while num_trials < max_num_trials:
            num_trials += 1
            # pick randomly a partition
            keys = list(partitions_dict.keys())
            picked_key = random.choice(keys)
            partition = partitions_dict[picked_key]
            # pick randomly a cluster in the partition
            cluster = random.choice(partition)
            # pick randomly another cluster2 that is close to cluster in the sense of the single-linkage distance
            other_clusters = copy.deepcopy(partition)
            other_clusters.remove(cluster)
            candidate_clusters = [c for c in other_clusters if single_linkage_distance(c, cluster) <= distance_threshold]
            if len(candidate_clusters) > 0:
                candidate_clusters_weights = [1 - single_linkage_distance(c, cluster) for c in candidate_clusters]
                cluster2 = random.choices(candidate_clusters, weights=candidate_clusters_weights)[0]
                # if the two clusters can be merged without exceeding the max size
                if len(cluster) + len(cluster2) <= max_cluster_size:
                    # create a new partition by merging these two clusters
                    other_clusters.remove(cluster2)
                    new_partition = other_clusters
                    merged_cluster = cluster + cluster2
                    merged_cluster.sort()
                    insert_pos = 0
                    found_insert_position = False
                    while found_insert_position == False and insert_pos < len(new_partition):
                        if merged_cluster[0] > new_partition[insert_pos][0]:
                            insert_pos += 1
                        else:
                            found_insert_position = True
                    new_partition.insert(insert_pos, merged_cluster)
                    # add the (new) partition to the dictionary
                    partitions_dict.update({make_name_from_partition(new_partition): new_partition})

        # return list of partitions generated
        list_partitions = list(partitions_dict.values())
        for i, partition in enumerate(list_partitions):
            print(str(i+1) + '. ', partition)
        return list_partitions

    def compact_pieces(self, pieces):
        """
        Compact pieces so that they fit into a tighter bounding polygon.
        :param packed_pieces: list of Shapely Polygons
        :return: list of Shapely Polygons
        """
        radius_step = 1
        # if pieces is passed as a MultiPolygon
        if str(type(pieces)) == '<class \'shapely.geometry.multipolygon.MultiPolygon\'>':
            # transform it into a list of Polygons
            pieces = list(pieces)
        # if there is only one piece in the list
        if len(pieces) <= 1:
            # return the list as it is
            return pieces
        _, min_area = self.get_bounding_polygon(pieces)
        directions = [Point([0, radius_step]),
                      Point([radius_step, 0]),
                      Point([-radius_step, 0]),
                      Point([0, -radius_step]),
                      Point([radius_step, radius_step]),
                      Point([-radius_step, radius_step]),
                      Point([-radius_step, -radius_step]),
                      Point([radius_step, -radius_step])]
        area_is_reduceable = True
        while area_is_reduceable == True:
            area_is_reduceable = False
            for i, piece in enumerate(pieces):
                best_area_improvement = 0
                best_configuration = None
                for direction in directions:
                    translated_piece = self.translate_shape(piece, direction)
                    other_pieces = copy.deepcopy(pieces)
                    del other_pieces[i]
                    if not self.is_intersection(translated_piece, other_pieces):
                        new_configuration = other_pieces
                        new_configuration.append(translated_piece)
                        _, new_area = QUANTUM_PACKER.get_bounding_polygon(new_configuration)
                        if new_area < min_area:
                            improvement = min_area - new_area
                            if improvement > best_area_improvement:
                                best_area_improvement = improvement
                                best_configuration = new_configuration
                                best_new_area = new_area
                if best_configuration is not None:
                    pieces = best_configuration
                    min_area = best_new_area
                    area_is_reduceable = True
                    break
        return pieces

    @staticmethod
    def solve_TSP_by_brute_force(D):
        """
        Solves the TSP problem (find shortest hamiltonian path) using brute force.
        Enumerates all paths or routes and finds the shortest one.
        :param D: square distance matrix as numpy array
        :return: shortest_path, minimum_length
        """
        n = D.shape[0]
        paths = list(itertools.permutations([i for i in range(n)]))
        shortest_path = []
        minimum_length = np.inf
        for path in paths:
            length = sum([D[path[i], path[i + 1]] for i in range(n - 1)])
            if length < minimum_length:
                minimum_length = length
                shortest_path = list(path)
        # return the shortest path and its length
        return shortest_path, minimum_length

    @staticmethod
    def solve_TSP_by_simulated_annealing(D):
        """
        Solves the TSP problem (find shortest hamiltonian path) using simulated annealing.
        :param D: square distance matrix as numpy array
        :return: shortest_path, minimum_length
        """
        # scale the matrix so that its maximum value is 1
        scaling_factor = np.ndarray.max(D)
        D = D / scaling_factor
        # number of cities
        num_cities = D.shape[0]
        # dimension of the quadratic matrix for the QUBO
        N = num_cities ** 2
        # initialize the quadratic matrix of the QUBO
        Q = np.zeros((N, N))
        # penalty coefficient
        A = np.max(D) + 1

        def get_qubit_index(u, i):
            """
            Maps a pair of integers (u, i) for visiting city u at tour i to a qubit index.
            :param u: city index
            :param i: tour index
            :return: qubit index = u + i * num_cities if i < num_cities, otherwise u
            """
            return u + (i % num_cities) * num_cities

        # compute the coefficients of the quadratic matrix
        for i in range(num_cities):
            for j in range(num_cities):
                for p in range(num_cities - 1):
                    q1 = get_qubit_index(i, p)
                    q2 = get_qubit_index(j, p+1)
                    Q[q1, q2] += D[i, j]
        for p in range(num_cities):
            for i in range(num_cities):
                q1 = get_qubit_index(i, p)
                Q[q1, q1] += -2 * A
                for j in range(num_cities):
                    q2 = get_qubit_index(j, p)
                    Q[q1, q2] += A
        for i in range(num_cities):
            for p in range(num_cities):
                q1 = get_qubit_index(i, p)
                Q[q1, q1] += -2 * A
                for q in range(num_cities):
                    q2 = get_qubit_index(i, q)
                    Q[q1, q2] += A

        # solve the QUBO using SA
        solver = QUBO_SA_SOLVER(Q)
        best_sample_dict = solver.solve(num_reads=1000).sample

        path_string = ''
        for i in range(N):
            key = 'x[' + str(i) + ']'
            value = best_sample_dict[key]
            path_string += str(value)
        shortest_path = TSP_QAOA_SOLVER.path_from_string(path_string)

        minimum_length = 0
        for i, j in zip(shortest_path[:-1], shortest_path[1:]):
            minimum_length += D[i, j]

        # return the shortest path and its length
        return shortest_path, minimum_length * scaling_factor

    def generate_partitions(self, P, R, num_generations=50):
        """

        :param P: partitions support (list of possible clusters)
        :param R: set of rectangles built from the clusters
        :param num_generations: number of times a random partition is generated
        :return:
        """

        def make_name_from_cluster(_c):
            c_str = [str(x) for x in _c]
            cluster_name = '-'.join(c_str)
            return cluster_name

        def make_name_from_partition(_partition):
            elements_names = []
            for e in _partition:
                cluster_name = make_name_from_cluster(e)
                elements_names.append(cluster_name)
            name = '_'.join(elements_names)
            return name

        print('Generating partitions:')
        generated_partitions = {}
        # compute for each cluster in P the complement of the waste percentage (geometric compatibility)
        cluster_geometric_compatibilities = {}
        for c in P:
            key = make_name_from_cluster(c)
            gi = R[key][3]
            cluster_geometric_compatibilities.update({key: 1 - gi})
        # compute a list of the cluster names
        cluster_names = list(cluster_geometric_compatibilities.keys())
        # for each random generation of a partition
        for _ in range(num_generations):
            # initialize the partition
            partition = []
            # initialize the set of pieces that can be selected for the partition
            selectable_pieces = [i for i in range(self.num_pieces)]
            while len(selectable_pieces) > 0:
                # determine the set of clusters composed only of selectable pieces
                selectable_clusters = []
                selectable_clusters_compatibilities = []
                for c in P:
                    is_selectable_cluster = True
                    for e in c:
                        if e not in selectable_pieces:
                            is_selectable_cluster = False
                    if is_selectable_cluster:
                        selectable_clusters.append(c)
                        cluster_name = make_name_from_cluster(c)
                        selectable_clusters_compatibilities.append(cluster_geometric_compatibilities[cluster_name])
                # pick a cluster from the top-3 compatibilities
                top_compatibilities = copy.deepcopy(selectable_clusters_compatibilities)
                top_compatibilities.sort(reverse=True)
                if len(top_compatibilities) >= 3:
                    compatibility_threshold = top_compatibilities[2]
                else:
                    compatibility_threshold = top_compatibilities[-1]
                best_clusters = [_c for _i, _c in enumerate(selectable_clusters) if selectable_clusters_compatibilities[_i] >= compatibility_threshold]
                c = random.choice(best_clusters)
                # add the cluster to the partition
                insert_pos = 0
                found_insert_position = False
                while found_insert_position == False and insert_pos < len(partition):
                    if c[0] > partition[insert_pos][0]:
                        insert_pos += 1
                    else:
                        found_insert_position = True
                partition.insert(insert_pos, c)
                # remove the pieces of the cluster from the selectable pieces
                selectable_pieces = [i for i in selectable_pieces if i not in c]
            # add the partition if it is new
            partition_name = make_name_from_partition(partition)
            if partition_name not in generated_partitions.keys():
                generated_partitions.update({partition_name: partition})
                print('    ', partition)
        # convert the dictionary to a list
        generated_partitions_list = list(generated_partitions.values())
        return generated_partitions_list

    def solve(self):
        start_time = time.perf_counter()
        # initialize the container
        L = 1e9
        H = self.H
        # container (as Polygon)
        container = Polygon([(0, 0), (L, 0), (L, H), (0, H)])
        # Compute distance and geometrical incompatibility matrices
        D, GI = self.compute_distance_matrix()
        distance_matrix_time = time.perf_counter()
        # Agglomerative clustering of the pieces (several clustering solutions are proposed) and
        # each solution defines then a possible partitioning of the set of pieces
        partitions = []
        all_gi = GI.flatten()
        all_gi.sort()
        gi_thresholds = list(set(all_gi.tolist()))
        gi_thresholds.sort()

        print('Partitioning the set of pieces using stochastic agglomerative clustering:')
        partitions = self.stochastic_agglomerative_clustering(GI, self.distance_threshold, self.max_cluster_size,
                                                              self.n_partitions)

        # Compute the partitions support, i.e. the set of sets of pieces that appear in the partitions
        partitions_support = []
        for partition in partitions:
            for set_of_pieces in partition:
                if set_of_pieces not in partitions_support:
                    partitions_support.append(set_of_pieces)

        # Solve the TSP problem for each set of pieces in partitions_support and store the hamiltonian paths
        num_TSP_instances = len(partitions_support)
        print('Solving ' + str(num_TSP_instances) + ' TSP problems...')

        # First, we build a list of unique TSPs to solve
        unique_set_of_pieces_keys = []
        problem_names = []
        problem_pieces = []
        distance_matrices = []
        for tsp_problem_index, P in enumerate(partitions_support):
            # key for the set of the pieces P
            set_of_pieces_key = '-'.join([str(i) for i in P])
            # modulo key for the set of pieces
            set_of_pieces_modulo_key = self.modulo_key(set_of_pieces_key)
            # if the modulo key is not in the list
            if set_of_pieces_modulo_key not in unique_set_of_pieces_keys:
                # add the TSP to the list
                unique_set_of_pieces_keys.append(set_of_pieces_modulo_key)
                D_P = D[P, :][:, P]
                distance_matrices.append(D_P)
                problem_names.append(set_of_pieces_modulo_key)
                problem_pieces.append(P)

        # Then we solve all the TSPs in the list using the chosen solver
        if self.TSP_solver == 'QAOA' or self.TSP_solver == 'QAOA+':
            print('Solving TSPs using QAOA...')
            batch_solver = TSP_BATCH_SOLVER_QAOA(problem_names,
                                                 distance_matrices,
                                                 hamiltonian_path=True,
                                                 use_approximate_optimization=self.use_approximate_optimization,
                                                 p=self.num_repeats,
                                                 num_shots=self.num_shots,
                                                 backend=self.backend,
                                                 solver_backend=self.solver_backend,
                                                 classical_optimizer=self.optimizer)
            best_hamiltonian_paths = batch_solver.run()
            # reassign the index of the pieces
            for key in best_hamiltonian_paths:
                pieces_indices = key.split('-')
                pieces_indices = [int(s) for s in pieces_indices]
                hamiltonian_path = best_hamiltonian_paths[key][0]
                path_length = best_hamiltonian_paths[key][0]
                reindexed_hamiltonian_path = [pieces_indices[i] for i in hamiltonian_path]
                best_hamiltonian_paths.update({key: [reindexed_hamiltonian_path, path_length]})
        # using simulated annealing
        elif self.TSP_solver == 'SA':
            print('Solving TSPs using simulated annealing...')
            best_hamiltonian_paths = {}
            for p, problem_name in enumerate(problem_names):
                D = distance_matrices[p]
                shortest_path, minimum_length = QUANTUM_PACKER.solve_TSP_by_simulated_annealing(D)
                best_hamiltonian_paths.update({problem_name: [shortest_path, minimum_length]})
        # using brute force
        else:
            print('Solving TSPs using brute force...')
            best_hamiltonian_paths = {}
            for p, problem_name in enumerate(problem_names):
                D = distance_matrices[p]
                P = problem_pieces[p]
                shortest_path, minimum_length = QUANTUM_PACKER.solve_TSP_by_brute_force(D)
                print('--- BRUTE FORCE TSP SOLVER ---')
                print('Pieces:')
                print(P)
                print('D=')
                print(D)
                print('shortest hamiltonian path=')
                solution = [P[i] for i in shortest_path]
                print(solution)
                print('minimum distance=')
                print(minimum_length)
                print('----------')
                print(minimum_length)
                best_hamiltonian_paths.update({problem_name: [solution, minimum_length]})

        TSP_time = time.perf_counter()

        # Store for all sets of pieces (key) the corresponding sequence of pieces to pack
        hamiltonian_paths_dict = {}
        for tsp_problem_index, P in enumerate(partitions_support):
            # key for the set of the pieces P
            set_of_pieces_key = '-'.join([str(i) for i in P])
            # modulo key for the set of pieces
            set_of_pieces_modulo_key = self.modulo_key(set_of_pieces_key)
            # modulo key for the set of pieces
            print('TSP problem #' + str(tsp_problem_index+1) + ' for set of pieces {' + set_of_pieces_key.replace('-', ',') + '} REF:' + set_of_pieces_modulo_key)
            # get the path already computed for the modulo key
            hamiltonian_path = best_hamiltonian_paths[set_of_pieces_modulo_key][0]
            # store the hamiltonian path found
            hamiltonian_paths_dict.update({set_of_pieces_key: hamiltonian_path})

        # Pack greedily all sets of pieces in rectangles in the order of the hamiltonian path or the reverse order
        # and store into the pool of rectangles R (dictionary) the solution with minimum area
        print('Packing pieces in ' + str(len(partitions_support)) + ' rectangles...')
        storage_file = 'rectangles_dictionary_' + self.instance_name + '.p'
        try:
            # load the dictionary
            R = pickle.load(open(storage_file, 'rb'))
        except:
            # create an empty dictionary
            R = {}
            # save the file
            pickle.dump(R, open(storage_file, 'wb'))
        for index_P, P in enumerate(partitions_support):
            # key for the set of the pieces P
            set_of_pieces_key = '-'.join([str(i) for i in P])
            # modulo key for the set of pieces
            set_of_pieces_modulo_key = self.modulo_key(set_of_pieces_key)
            print('Packing pieces ' + set_of_pieces_key)
            # if the same set of pieces has already been packed
            if set_of_pieces_modulo_key in R.keys():
                # retrieve the result of the packing
                (packed_pieces, bbox, dimensions, waste_pc) = R[set_of_pieces_modulo_key]
                # reuse the pre-computed result and store in the dictionary
                R.update({set_of_pieces_key: (packed_pieces, bbox, dimensions, waste_pc)})
            else:
                # get the hamiltonian path that was obtained as solution of the TSP
                ordered_piece_indices_to_pack = hamiltonian_paths_dict[set_of_pieces_key]
                # pack greedily in the order of the hamiltonian path and compact the pieces
                packed_pieces = self.greedy_packer(ordered_piece_indices_to_pack)
                packed_pieces = self.compact_pieces(packed_pieces)
                # pack greedily in the reverse order of the hamiltonian path and compact the pieces
                reverse_path = copy.deepcopy(ordered_piece_indices_to_pack)
                reverse_path.reverse()
                packed_pieces_reverse = self.greedy_packer(reverse_path)
                packed_pieces_reverse = self.compact_pieces(packed_pieces_reverse)
                # fit bounding polygons to the packed pieces
                _, bounding_polygon_area = QUANTUM_PACKER.get_bounding_polygon(packed_pieces)
                _, bounding_polygon_reverse_area = QUANTUM_PACKER.get_bounding_polygon(packed_pieces_reverse)
                # Get bounding boxes and dimensions
                bbox, _ = self.fit_bounding_box(packed_pieces)
                bbox_reverse, _ = self.fit_bounding_box(packed_pieces_reverse)
                dimensions = self.get_bounding_box_dimensions(bbox)
                dimensions_reverse = self.get_bounding_box_dimensions(bbox_reverse)
                # store the packed pieces and bounding box for the bounding polygon with smallest area with the key and also the modulo key
                if bounding_polygon_area <= bounding_polygon_reverse_area:
                    best_packing_order = ordered_piece_indices_to_pack
                    waste_pc = (bbox.area - sum([p.area for p in packed_pieces])) / bbox.area
                    R.update({set_of_pieces_modulo_key: (packed_pieces, bbox, dimensions, waste_pc)})
                    R.update({set_of_pieces_key: (packed_pieces, bbox, dimensions, waste_pc)})
                    print('Showing the result of greedy packer for: ', best_packing_order)
                    rect_title = 'R({' + set_of_pieces_key.replace('-', ',') + '}) seq. [' + ','.join([str(i) for i in best_packing_order]) + '] dims=' + "{:.1f}".format(dimensions[0]) + ' x ' + "{:.1f}".format(dimensions[1]) + ' W=' + "{:.1f}".format(waste_pc * 100)  + '%'
                    self.show_layout(packed_pieces,
                                     rect_title,
                                     rectangles_dict={set_of_pieces_key: (packed_pieces, bbox, dimensions)},
                                     hide_container=True)
                else:
                    best_packing_order = reverse_path
                    waste_pc = (bbox_reverse.area - sum([p.area for p in packed_pieces_reverse])) / bbox_reverse.area
                    R.update({set_of_pieces_modulo_key: (packed_pieces_reverse, bbox_reverse, dimensions_reverse, waste_pc)})
                    R.update({set_of_pieces_key: (packed_pieces_reverse, bbox_reverse, dimensions_reverse, waste_pc)})
                    print('Showing the result of greedy packer for: ', best_packing_order)
                    rect_title_reverse = 'R({' + set_of_pieces_key.replace('-', ',') + '}) + seq. [' + ','.join([str(i) for i in best_packing_order]) + '] dims=' + "{:.1f}".format(dimensions_reverse[0]) + ' x ' + "{:.1f}".format(dimensions_reverse[1]) + ' W=' + "{:.1f}".format(waste_pc * 100) + '%'
                    self.show_layout(packed_pieces_reverse,
                                     rect_title_reverse,
                                     rectangles_dict={
                                         set_of_pieces_key: (packed_pieces_reverse, bbox_reverse, dimensions_reverse)},
                                     hide_container=True)
            # save the file every 5 rectangles computed
            if index_P % 5 == 0:
                pickle.dump(R, open(storage_file, 'wb'))
        # save the file with all rectangles computed
        pickle.dump(R, open(storage_file, 'wb'))
        rectangles_time = time.perf_counter()

        # Extract the list of all clusters which we have packed into a rectangle and saved
        all_partitions_support = []
        all_partitions_support_str = list(R.keys())
        for s in all_partitions_support_str:
            cluster = [int(i) for i in s.split('-')]
            all_partitions_support.append(cluster)

        # Generate candidate partitions based on all existing partitions support and the formed rectangles
        partitions = self.generate_partitions(all_partitions_support, R)

        # Initialize the best layout in case none is found that fits the bin size
        L_star = 1e9
        best_layout_found = None
        best_rectangle_layout_found = None
        best_partition = ''
        # For each partition of the set of pieces (trial)
        print('===============')
        for t, partition in enumerate(partitions):
            print('Partition ' + str(t + 1))
            print(partition)

            # build the rectangle names (modulo key of the set of pieces)
            rectangle_names = []
            for P in partition:
                # key for the set of the pieces e
                set_of_pieces_key = '-'.join([str(i) for i in P])
                # append the name of the rectangle, made up from the set of pieces key
                rectangle_names.append(set_of_pieces_key)

            # build the list of rectangles containing the packed pieces
            rectangles = []
            for rectangle_name in rectangle_names:
                # make sure the dimensions are integers
                l = math.ceil(R[rectangle_name][2][0])
                h = math.ceil(R[rectangle_name][2][1])
                rectangles.append((l, h))

            # try to pack the rectangles inside the regular container with current best length or enlarged versions
            for i in range(self.num_relaxations):
                length_limit = L_star * (1 + self.alpha * i / 100)
                for j in range(self.num_relaxations):
                    height_limit = H * (1 + self.beta * j / 100)
                    bin = (length_limit, height_limit)
                    print('        packing rectangles in strip of height H=' + str(height_limit))
                    rectangles_arrangement = self.pack_rectangles(rectangles, rectangle_names, bin)
                    # if the packing of rectangles is successful in that bin
                    if rectangles_arrangement is not None:
                        print('            found a layout that fits in that bin')
                        rectangle_packing_successful = True
                        layout, rectangles_layout = self.unpack_rectangles(rectangles_arrangement, R)
                        layout = self.local_optimization(layout, container)
                        h = self.required_height(layout)
                        l = self.required_bin_length(layout)
                        # if the optimized layout fits the imposed height
                        if h <= H:
                            print('            the optimized layout fits the container height')
                            # Update the best layout if the length is reduced
                            if l < L_star:
                                L_star = l
                                best_layout_found = layout
                                best_rectangle_layout_found = rectangles_layout
                                best_partition = partition
                                print('IMPROVED CONTAINER LENGTH: ' + str(L_star))
                                print(rectangles_arrangement)
                            else:
                                print('            the container length has not been improved')

        # final optimization by relocating the rightmost piece
        final_optimization_time = time.perf_counter()
        print('Running final optimization...')
        success = True
        while success:
            best_layout_found, moved_piece_index, success = self.relocate_rightmost_piece(best_layout_found)
            L_star = self.required_bin_length(best_layout_found)

        end_time = time.perf_counter()

        num_QUBOs_solved = len(problem_names)
        layout_area = MultiPolygon(best_layout_found).area
        minimum_sheet_area = L_star * self.H
        wasted_surface = minimum_sheet_area - layout_area
        waste_percentage = wasted_surface / minimum_sheet_area
        # print results
        print('===============')
        print('Problem instance: ' + self.instance_name)
        print('Number of rotations: ' + str(self.n_rotations))
        print('Distance matrix and Nofit Functions: ' + "{:.2f}".format(distance_matrix_time - start_time) + 's')
        print(str(num_TSP_instances) + ' TSPs (' + str(num_QUBOs_solved) + ' QUBOs): ' + "{:.2f}".format(TSP_time - distance_matrix_time) + 's')
        print('Greedy packer: ' + "{:.2f}".format(rectangles_time - TSP_time) + 's')
        print('Rectangular packer and local optimization: ' + "{:.2f}".format(final_optimization_time - rectangles_time) + 's')
        print('Final optimization: ' + "{:.2f}".format(end_time - final_optimization_time) + 's')
        print('Total solver time: ' + "{:.2f}".format(end_time - start_time) + 's')
        print('Best layout found requires a length of ' + "{:.2f}".format(L_star))
        print('Layout area: ' + "{:.2f}".format(layout_area))
        print('Minimum sheet area: ' + "{:.2f}".format(minimum_sheet_area))
        print('Wasted area: ' + "{:.2f}".format(wasted_surface))
        print('Waste percentage: ' + "{:.2f}".format(100*waste_percentage) + '%')
        print('Ordered partition used: ')
        print(best_partition)
        print('===============')
        # return the best layout found (with minimum length)
        return best_layout_found, L_star, best_partition, best_rectangle_layout_found

    def show_layout(self, layout, title, board_dimensions=None, rectangles_dict=None, hide_container=False):
        # create a new plot
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        if hide_container==False:
            # plot the board
            if board_dimensions == None:
                W = self.board_dimensions[0]
                H = self.board_dimensions[1]
            else:
                W = board_dimensions[0]
                H = board_dimensions[1]
            board_color = 'gray'
            board = Polygon([(0, 0), (W, 0), (W, H), (0, H)])
            xs, ys = board.exterior.xy
            axs.fill(xs, ys, alpha=1.0, fc=board_color, ec='black')
        # plot the rectangles
        if rectangles_dict is not None:
            for rectangle_name in rectangles_dict.keys():
                (packed_pieces, bounding_box, dimensions) = rectangles_dict[rectangle_name]
                my_rect = bounding_box
                xs, ys = my_rect.exterior.xy
                axs.fill(xs, ys, alpha=1.0, fc='grey', ec='black')
                bbox_center = self.center_of_shape(my_rect)
                #axs.text(bbox_center.x, bbox_center.y, 'R(' + rectangle_name.replace('-', ',') + ')')
        # plot the pieces in white
        for i, piece in enumerate(layout):
            xs, ys = piece.exterior.xy
            axs.fill(xs, ys, alpha=1.0, fc='white', ec='black')
        plt.title(title)
        plt.show()
        file_name = title + '.pdf'
        fig.savefig(file_name)

