import random
from shapely.ops import voronoi_diagram, cascaded_union
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely import affinity
from shapely.validation import make_valid
import pickle
import math
import matplotlib.pyplot as plt


class PROBLEM_GENERATOR:

    def __init__(self, instance_name, min_length, max_length, num_rectangles, max_num_polygons, rotations):
        """"

        """
        self.instance_name = instance_name
        self.Phi = rotations

        # Generate a rectangular container with sides comprised between min_length and max_length
        self.H = random.randint(int(min_length), int(max_length))
        self.L = random.randint(int(min_length), int(max_length))
        print('Generating container with dimensions H=' + str(self.H) + ' x L=' + str(self.L))

        # Split into rectangles num_rectangles
        rectangles = [Polygon([(0, 0), (self.L, 0), (self.L, self.H), (0, self.H)])]
        while len(rectangles) < num_rectangles:
            # pick a rectangle at random
            i = random.randint(0, len(rectangles) - 1)
            rectangle = rectangles[i]
            # pick a split direction
            if random.randint(0, 1) == 0:
                direction = 'horizontal'
            else:
                direction = 'vertical'
            # split the rectangle in that direction
            bounds = rectangle.bounds
            x_min = bounds[0]
            y_min = bounds[1]
            x_max = bounds[2]
            y_max = bounds[3]
            if direction == 'horizontal':
                y_split = random.randint(int(y_min + 0.1 * (y_max - y_min)), int(y_max - 0.1 * (y_max - y_min)))
                sub_rectangle1 = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_split), (x_min, y_split)])
                sub_rectangle2 = Polygon([(x_min, y_split+1), (x_max, y_split+1), (x_max, y_max), (x_min, y_max)])
            else:
                x_split = random.randint(int(x_min + 0.1 * (x_max - x_min)), int(x_max - 0.1 * (x_max - x_min)))
                sub_rectangle1 = Polygon([(x_min, y_min), (x_split, y_min), (x_split, y_max), (x_min, y_max)])
                sub_rectangle2 = Polygon([(x_split+1, y_min), (x_max, y_min), (x_max, y_max), (x_split+1, y_max)])
            # remove the rectangle and replace it by the new ones
            rectangles.append(sub_rectangle1)
            rectangles.append(sub_rectangle2)
            del rectangles[i]

        # Store the rectangles layout
        self.rectangles_layout = rectangles

        # Split each rectangle into 2 up to num_polygons polygons
        list_of_polygons = []
        for rectangle in rectangles:
            num_polygons = random.randint(2, max_num_polygons)
            if num_polygons == 1:
                list_of_polygons.append(rectangle)
            else:
                split_polygons = PROBLEM_GENERATOR.split_rectangle_into_polygons(rectangle, num_polygons)
                list_of_polygons += split_polygons

        # store the perfect layout
        self.layout = list_of_polygons

        # compute the waste of the layout
        self.area_container = self.H * self.L
        self.area_pieces = sum([polygon.area for polygon in list_of_polygons])
        self.waste_area = self.area_container - self.area_pieces
        self.waste_percentage = self.waste_area / self.area_container

        # show the perfect layout
        title = self.instance_name + ': H=' + str(self.H) + ', L=' + str(self.L) + ', W=' + str(
            "{:.2f}".format(self.waste_percentage * 100)) + '%'
        self.show_layout(self.layout, title, self.instance_name, board_dimensions=(self.L, self.H))

        # Shuffle the polygons
        random.shuffle(list_of_polygons)

        # Add random rotations
        pieces = []
        for piece in list_of_polygons:
            rotation_angle = random.choice(self.Phi)
            piece = PROBLEM_GENERATOR.rotate_shape(piece, rotation_angle)
            pieces.append(piece)

        # Store the polygons as pieces
        self.pieces = pieces

        # Saves the object
        storage_file = 'saved_packing_problem_' + self.instance_name + '.p'
        pickle.dump(self, open(storage_file, 'wb'))
        print('Saved problem instance ' + self.instance_name)

        # Print in the console the pieces
        print('Set of pieces:')
        for p, piece in enumerate(self.pieces):
            vertices = list(zip(*piece.exterior.coords.xy))
            ref_point = vertices[0]
            piece_str = 'P' + str(p) + ' [' + ', '.join(['(' + str(int(v[0] - ref_point[0])) + ', ' +
                                                         str(int(v[1] - ref_point[1])) + ')' for v in vertices]) + ']'
            print(piece_str)

    @staticmethod
    def randomize_segment(A, B, num_intermediate_points, orthogonal_deviation):
        """
        Creates a list of points joining A to B
        :param A: (x_A, y_A)
        :param B: (x_B, y_B)
        :return:
        """
        points = [A]
        AB = (B[0]-A[0], B[1]-A[1])
        L_AB = math.sqrt((AB[0])**2 + (AB[1])**2)
        u_AB = (AB[0] / L_AB, AB[1] / L_AB)
        v_AB = (-u_AB[1], u_AB[0])
        u_coords = []
        v_coords = []
        for _ in range(num_intermediate_points):
            u_coords.append(random.randint(0, int(math.floor(L_AB))))
            v_coords.append(random.randint(-int(orthogonal_deviation), int(orthogonal_deviation)))
        u_coords.sort()
        v_coords.sort()
        for i in range(num_intermediate_points):
            points.append((A[0] + u_coords[i] * u_AB[0] + v_coords[i] * v_AB[0], A[1] + u_coords[i] * u_AB[1] + v_coords[i] * v_AB[1]))
        points.append(B)
        return points

    @staticmethod
    def clip(x, a, b):
        if x < a:
            return a
        elif x > b:
            return b
        else:
            return x

    @staticmethod
    def distance(p, q):
        return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)

    @staticmethod
    def split_rectangle_into_polygons(rectangle, num_polygons, add_noise=False):
        """
        Splits a rectangle into n polygons.
        :param rectangle: a rectangle as shapely polygon
        :param num_polygons: number of polygons
        :return: list of Shapely polygons
        """
        bounds = rectangle.bounds
        x_min = bounds[0]
        y_min = bounds[1]
        x_max = bounds[2]
        y_max = bounds[3]
        # direction of alignement of the polygons
        if (x_max - x_min) >= (y_max - y_min):
            alignment_direction = 'horizontal'
        else:
            alignment_direction = 'vertical'
        # split the rectangle into a chain of num_polygons rectangular compartments in the alignment direction
        if alignment_direction == 'horizontal':
            split_positions = [x_min] + [random.randint(int(x_min + 0.1 * (x_max-x_min)),
                                                        int(x_max - 0.1 * (x_max-x_min))) for _ in range(num_polygons-1)] + [x_max]
        else:
            split_positions = [y_min] + [random.randint(int(y_min + 0.1 * (y_max-y_min)),
                                                        int(y_max - 0.1 * (y_max-y_min))) for _ in range(num_polygons-1)] + [y_max]
        split_positions = list(set(split_positions))
        split_positions.sort()
        # create borders separating the polygons
        borders = []
        for s, split in enumerate(split_positions):
            if alignment_direction == 'horizontal':
                A = (split, y_min)
                B = (split, y_max)
            else:
                A = (x_min, split)
                B = (x_max, split)
            num_intermediate_points = random.randint(1, 10)
            if (s > 0) and (s < len(split_positions) - 1):
                orthogonal_deviation = min(abs(split_positions[s+1] - split_positions[s]) / 3, abs(split_positions[s] - split_positions[s-1]) / 3)
            elif s == 0:
                orthogonal_deviation = abs(split_positions[s+1] - split_positions[s]) / 5
            else:
                orthogonal_deviation = abs(split_positions[s] - split_positions[s - 1]) / 5
            orthogonal_deviation = int(orthogonal_deviation)
            border = PROBLEM_GENERATOR.randomize_segment(A, B, num_intermediate_points, orthogonal_deviation)
            borders.append(border)
        # generate a polygon inside each compartment
        polygons = []
        for n in range(num_polygons):
            # generate a polygon from its two borders
            border1 = borders[n]
            border2 = borders[n+1]
            d_simple = PROBLEM_GENERATOR.distance(border1[-1], border2[0])
            d_reverse = PROBLEM_GENERATOR.distance(border1[-1], border2[-1])
            if d_simple > d_reverse:
                border2.reverse()
            points_list = border1 + border2
            polygon = Polygon(points_list)
            if polygon.is_valid == False:
                pass
            # make sure the polygon does not go outside the rectangle
            vertices = list(zip(*polygon.exterior.coords.xy))
            new_vertices = []
            for v in vertices:
                new_v = Point(PROBLEM_GENERATOR.clip(v[0], x_min, x_max), PROBLEM_GENERATOR.clip(v[1], y_min, y_max))
                new_vertices.append(new_v)
            polygon = Polygon(new_vertices)
            # add the polygon
            polygons.append(polygon)
        # simplify the polygon by removing vertices that are too close to each other
        simplified_polygons = []
        for i, polygon in enumerate(polygons):
            # list of vertices
            vertices = list(zip(*polygon.exterior.coords.xy))
            # remove duplicates
            vertices = list(set(vertices))
            # new list of vertices
            new_vertices = [vertices[0]]
            last_vertex = vertices[0]
            for j in range(1, len(vertices) - 1):
                # distances between the current vertex and its predecessor and successor
                current_vertex = vertices[j]
                d = PROBLEM_GENERATOR.distance(last_vertex, current_vertex)
                # if the distance is not too small
                if d >= polygon.length / 20:
                    # add the vertex
                    new_vertices.append(current_vertex)
                    last_vertex = current_vertex
            # rebuild the polygon and add to list
            polygon = Polygon(new_vertices)
            simplified_polygons.append(polygon)
        polygons = simplified_polygons
        # add noise to the position of the vertices of polygons, so as increase waste
        if add_noise:
            for i, polygon in enumerate(polygons):
                # fix amplitude of perturbations
                bounds = polygon.bounds
                x_min = bounds[0]
                y_min = bounds[1]
                x_max = bounds[2]
                y_max = bounds[3]
                sigma_x = int(0.05 * (x_max - x_min))
                sigma_y = int(0.05 * (y_max - y_min))
                # coordinates of the vertices of the polygon
                vertices = list(zip(*polygon.exterior.coords.xy))
                # create new polygon from a list new list of vertices
                new_vertices = []
                for v, vertex in enumerate(vertices):
                    new_vertex_valid = False
                    max_trials = 20
                    num_trials = 0
                    while (new_vertex_valid == False) and (num_trials <= max_trials):
                        # add random perturbation
                        delta_x = random.randint(-sigma_x, sigma_x)
                        delta_y = random.randint(-sigma_y, sigma_y)
                        new_vertex = (vertex[0] + delta_x, vertex[1] + delta_y)
                        # check if new vertex is in the original polygon
                        if polygon.contains(Point(new_vertex[0], new_vertex[1])) == True:
                            new_polygon = Polygon(new_vertices + [new_vertex] + vertices[v+1:])
                            # if new polygon defines a valid new polygon
                            if new_polygon.is_valid:
                                new_vertex_valid = True
                        # increase trials count
                        num_trials += 1
                    # add new vertex or keep old one
                    if new_vertex_valid == True:
                        new_vertices.append(new_vertex)
                    else:
                        new_vertices.append(vertex)
                new_polygon = Polygon(new_vertices)
                # replace by the new polygon
                del polygons[i]
                polygons.append(new_polygon)
        return polygons

    @staticmethod
    def center_of_shape(shape):
        """
        Centroid of the shape.
        :param shape: Shapely geometry
        :return: a Point
        """
        center = shape.centroid
        return Point(center.x, center.y)

    @staticmethod
    def rotate_shape(shape, rotation_angle, rotation_center=None):
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
            rotation_center = PROBLEM_GENERATOR.center_of_shape(shape)
        try:
            rotated_shape = affinity.rotate(shape, rotation_angle, rotation_center)
        except:
            pass
        return rotated_shape

    def show_layout(self, layout, title, file_name, board_dimensions=None, rectangles_dict=None, hide_container=False):
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
        file_name = file_name + '.pdf'
        fig.savefig(file_name)