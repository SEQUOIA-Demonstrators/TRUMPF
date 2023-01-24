from TSP_QAOA_SOLVER import *


class TSP_BATCH_SOLVER_QAOA:

    def __init__(self, problem_names,
                 distance_matrices,
                 hamiltonian_path,
                 use_approximate_optimization,
                 p,
                 num_shots,
                 backend,
                 solver_backend,
                 classical_optimizer):
        """
        Solves a batch of TSP problems using QAOA.
        :param problem_names: list of strings for the identifier names of the problems to solve
        :param distance_matrices: list of distance matrices as numpy array, same length as problem_names
        :param hamitonian_path: boolean, set to False by default if you want to find the shortest Hamitonian tour,
                                otherwise when set to True, computes the shortest Hamiltonian path (no return to the
                                first city)
        :param use_approximate_optimization: if True then uses the Quantum Approximate Optimization Algorithm, else uses
                                             the Quantum Alternate Operator Ansatz algorithm
        :param p: number of repetitions of unitaries for the QAOA algorithm
        :param num_shots: number of shots used when running the QAOA circuit
        :param backend: quantum backend where the QAOA circuits are optimized
        :param solver_backend: quantum backend where the optimized QAOA circuits are run to solve the TSPs
        :param classical_optimizer: optimizer used to tune the QAOA circuit parameters, for instance 'COBYLA', 'SPSA',
        'SLSQP', 'BFGS', 'L-BFGS-B'
        """
        self.problem_names = problem_names
        self.distance_matrices = distance_matrices
        self.hamiltonian_path = hamiltonian_path
        self.use_approximate_optimization = use_approximate_optimization
        self.p = p
        self.num_shots = num_shots
        self.backend = backend
        self.solver_backend = solver_backend
        self.classical_optimizer = classical_optimizer

    @staticmethod
    def compute_path_length(path, distance_matrix):
        """
        Computes the length of a path
        :param path: list of cities
        :return: total distance
        """
        length = 0
        for i, j in zip(path[:-1], path[1:]):
            length += distance_matrix[i, j]
        return length

    def run(self):
        """

        :return: dictionary with problem names as keys and [path, length] as values
        """

        # initialize the dictionary of TSP solutions
        tsp_solutions = {}

        # First, we build the list of all quantum circuits to execute
        print('Creating a list of ' + str(len(self.problem_names)) + ' QAOA circuits')
        quantum_circuits = []
        for i, problem_name in enumerate(self.problem_names):
            distance_matrix = self.distance_matrices[i]
            qaoa_instance = TSP_QAOA_SOLVER(problem_name,
                                           distance_matrix,
                                           hamiltonian_path=self.hamiltonian_path,
                                           use_approximate_optimization=self.use_approximate_optimization,
                                           p=self.p,
                                           num_shots=self.num_shots,
                                           backend=self.backend,
                                           classical_optimizer=self.classical_optimizer)
            qaoa_circuit = qaoa_instance.get_optimized_qaoa_circuit()
            quantum_circuits.append(qaoa_circuit)

        # Then, we execute all the circuits on the specified solver backend
        print('Executing the list of quantum circuits')
        all_results_list = execute(quantum_circuits, self.solver_backend, shots=self.num_shots).result()
        # for the result of each circuit
        for p in range(len(self.problem_names)):
            # name of the tsp we have solved
            tsp_name = self.problem_names[p]
            # distance matrix
            distance_matrix = self.distance_matrices[p]
            # get the dictionary of counts
            counts_dict = all_results_list.get_counts(p)
            # keeps the results with highest number of counts
            best_shots_dict = TSP_QAOA_SOLVER.filter_highest_counts(counts_dict, max_rank=1)
            # show the histogram
            TSP_QAOA_SOLVER.show_histogram(counts_dict, best_shots_dict, tsp_name)
            # keep the path with minimum distance from the results
            shortest_path = []
            minimum_length = np.inf
            for path_string in best_shots_dict.keys():
                path = TSP_QAOA_SOLVER.path_from_string(path_string)
                path_length = TSP_BATCH_SOLVER_QAOA.compute_path_length(path, distance_matrix)
                if path_length < minimum_length:
                    minimum_length = path_length
                    shortest_path = path
            # store the solution to this TSP
            tsp_solutions.update({tsp_name: [shortest_path, minimum_length]})
        return tsp_solutions


