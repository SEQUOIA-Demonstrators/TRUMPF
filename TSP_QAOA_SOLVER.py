import qiskit as qk
from qiskit import execute
import numpy as np
import math
from scipy.optimize import minimize
from qiskit.algorithms.optimizers import SPSA
import itertools
import time
import random
import matplotlib.pyplot as plt


class TSP_QAOA_SOLVER:

    def __init__(self, problem_name, distance_matrix, hamiltonian_path=False, use_approximate_optimization=True, p=3, num_shots=512,
                 backend=None, classical_optimizer='SLSQP'):
        """
        Class constructor.
        :param problem_name: string to refer to the TSP problem
        :param distance_matrix: distance matrix as numpy array
        :param hamitonian_path: boolean, set to False by default if you want to find the shortest Hamitonian tour,
                                otherwise when set to True, computes the shortest Hamiltonian path (no return to the
                                first city)
        :param use_approximate_optimization: if True then uses the Quantum Approximate Optimization Algorithm, else uses
                                             the Quantum Alternate Operator Ansatz algorithm
        :param p: number of repetitions of unitaries for the QAOA algorithm
        :param num_shots: number of shots used when running the QAOA circuit
        :param backend: quantum backend where the QAOA circuit is optimized and run to solve the TSP
        :param classical_optimizer: optimizer used to tune the QAOA circuit parameters, for instance 'COBYLA', 'SPSA',
        'SLSQP', 'BFGS', 'L-BFGS-B'
        """
        self.problem_name = problem_name
        self.scaling_factor = np.amax(distance_matrix)
        self.distance_matrix = distance_matrix / self.scaling_factor
        self.num_cities = distance_matrix.shape[0]
        self.n_qubits = self.num_cities ** 2
        self.hamiltonian_path = hamiltonian_path
        if hamiltonian_path:
            # for a Hamiltonian path, stop before tour step i = num_cities-1
            self.sum_stop = self.num_cities-1
        else:
            # for a Hamiltonian tour, stop before tour step i = num_cities
            self.sum_stop = self.num_cities
        if use_approximate_optimization:
            self.algorithm = 'Quantum Approximate Optimization Algorithm'
            self.mixer = 'transverse field'
        else:
            self.algorithm = 'Quantum Alternating Operator Ansatz'
            self.mixer = 'swap mixer'
        self.p = p
        self.num_shots = num_shots
        self.optimizer_backend = backend
        self.solver_backend = backend
        self.classical_optimizer = classical_optimizer

    def get_qubit_index(self, u, i):
        """
        Maps a pair of integers (u, i) for visiting city u at tour i to a qubit index.
        :param u: city index
        :param i: tour index
        :return: qubit index = u + i * num_cities if i < num_cities, otherwise u
        """
        return u + (i % self.num_cities) * self.num_cities

    @staticmethod
    def get_city_and_tour_from_qubit(q, num_cities):
        """
        Inverse function of get_qubit_index
        :param q: qubit index
        :return: city index, tour index
        """
        u = q % num_cities
        i = int(math.floor(q / num_cities))
        return u, i

    def add_phase_separator(self, qc, _gamma):
        """
        Adds the phase separator to a circuit.
        :param qc: quantum circuit
        :param _gamma: scalar parameter gamma for the phase separator to add
        :return: quantum circuit expanded with e^{-i*_gamma*H_P^{(enc)}} where
                 H_P^{(enc)} = \sum_i \sum_u \sum_v d(u,v) * Z_{u,i} * Z_{v,j+1}
        """
        # for each tour step i
        for i in range(self.sum_stop):
            # for each city u
            for u in range(self.num_cities):
                # for each city v
                for v in range(self.num_cities):
                    # adds gate e^{-i*gamma*d(u,v)*Z_{u,i}*Z_{v,i+1}}
                    qubit_u_i = self.get_qubit_index(u, i)
                    qubit_v_iplus1 = self.get_qubit_index(v, i+1)
                    qc.rzz(2 * _gamma * self.distance_matrix[u, v], qubit_u_i, qubit_v_iplus1)
        # return quantum circuit with one application of the phase separator
        return qc

    def C3RXGate(self, theta):
        qc = qk.QuantumCircuit(4)
        qc.h(3)
        qc.p(theta / 8, [0, 1, 2, 3])
        qc.cx(0, 1)
        qc.p(-theta / 8, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.p(-theta / 8, 2)
        qc.cx(0, 2)
        qc.p(theta / 8, 2)
        qc.cx(1, 2)
        qc.p(-theta / 8, 2)
        qc.cx(0, 2)
        qc.cx(2, 3)
        qc.p(-theta / 8, 3)
        qc.cx(1, 3)
        qc.p(theta / 8, 3)
        qc.cx(2, 3)
        qc.p(-theta / 8, 3)
        qc.cx(0, 3)
        qc.p(theta / 8, 3)
        qc.cx(2, 3)
        qc.p(-theta / 8, 3)
        qc.cx(1, 3)
        qc.p(theta / 8, 3)
        qc.cx(2, 3)
        qc.p(-theta / 8, 3)
        qc.cx(0, 3)
        qc.h(3)
        return qc.to_gate()

    def four_qubit_swap_gate(self, parameter):
        # TODO: This gate could probably be optimized further
        gate = qk.QuantumCircuit(4)
        gate.cx(2, 3)
        gate.cx(2, 0)
        gate.cx(1, 2)
        gate.x(2)
        gate.append(self.C3RXGate(theta=2 * parameter), (0, 2, 3, 1))
        gate.x(2)
        gate.cx(1, 2)
        gate.cx(2, 0)
        gate.cx(2, 3)
        transpiled = qk.transpile(gate, optimization_level=3)
        return transpiled.to_gate()

    def add_four_qubit_swap(self, qc, beta, u, v, i):
        """
        Add to the circuit qc the i-th adjacent value-selective swap partial mixer
        which swaps cities u and v between tour positions i and i+1.
        :param qc: quantum circuit to which to add a partial swap that swaps cities u and v between tour positions i and i+1
        :param beta: scalar parameter beta for the mixer
        :param u: city index
        :param v: city index
        :param i: tour position
        :return: quantum circuit expanded with e^{-i*beta*H_{PS, i, {u,v}}^{(enc)}}
        """
        qubit_u_i = self.get_qubit_index(u, i)
        qubit_u_iplus1 = self.get_qubit_index(u, i+1)
        qubit_v_i = self.get_qubit_index(v, i)
        qubit_v_iplus1 = self.get_qubit_index(v, i+1)
        swap_gate = self.four_qubit_swap_gate(beta)
        qc.append(swap_gate, (qubit_u_i, qubit_v_i, qubit_u_iplus1, qubit_v_iplus1))
        return qc

    def add_swap_mixer(self, qc, beta):
        """
        Partial swap mixer used for a single QAOA iteration, hence only a single parameter is used.
        """
        for i in range(self.sum_stop):
            for u in range(self.num_cities):
                for v in range(self.num_cities):
                    if u != v:
                        qc = self.add_four_qubit_swap(qc, beta, u, v, i)
        return qc

    def add_transverse_field_mixer(self, qc, beta):
        """
        Mixer based on the transverse field Hamiltonian used in Approximate Optimization for one iteration,
        hence only a single parameter is used.
        :param qc:
        :param beta:
        :return:
        """
        for i in range(self.n_qubits):
            qc.rx(beta, i)
        return qc

    def create_qaoa_circuit(self):
        """
        Creates a parametrized QAOA circuit with p repetitions for solving the TSP problem.
        :return: quantum circuit
        """
        beta = [qk.circuit.Parameter("beta{}".format(i)) for i in range(self.p)]
        gamma = [qk.circuit.Parameter("gamma{}".format(i)) for i in range(self.p)]
        # initialize quantum circuit with n qubits
        qc = qk.QuantumCircuit(self.n_qubits)
        # initial state is the path 0, 1, 2, ... (visit city i at round i)
        for i in range(0, self.n_qubits, self.num_cities + 1):
            qc.x(i)
        # for each repetition of the unitaries
        for i in range(self.p):
            # add the phase separator and the mixer
            qc = self.add_phase_separator(qc, gamma[i])
            if self.mixer == 'swap mixer':
                qc = self.add_swap_mixer(qc, beta[i])
            else:
                qc = self.add_transverse_field_mixer(qc, beta[i])
        # measure all the qubits
        qc.measure_all()
        # return the transpiled circuit and its parameters
        return qk.transpile(qc, optimization_level=3), beta, gamma

    @staticmethod
    def is_valid_path(string):
        """
        Check if a bitstring represents a valid path.
        :param string: bitstring of length num_qubits
        :return: boolean, X array
        """
        n_qubits = len(string)
        # check that n_qubits is a perfect square
        if n_qubits != math.isqrt(n_qubits) ** 2:
            print('ERROR: n_qubits in string ' + string + ' is not a perfect square')
        num_cities = int(math.sqrt(n_qubits))
        # initialize matrix ((x_{i,p}))
        X = np.zeros((num_cities, num_cities))
        for q in range(n_qubits):
            qubit_value = string[q]
            if qubit_value == '1':
                # we visit city i at tour step p
                i, p = TSP_QAOA_SOLVER.get_city_and_tour_from_qubit(q, num_cities)
                X[i, p] = 1
        for p in range(num_cities):
            if np.sum(X[:, p]) != 1.0:
                return False, X
        for i in range(num_cities):
            if np.sum(X[i, :]) != 1.0:
                return False, X
        return True, X

    @staticmethod
    def correct_path(X):
        """
        Corrects a string representing an invalid path through the cities.
        :param X: matrix ((x_{i,p})) that tells if we visit city i and tour step p
        :return: bitstring representing a valid path
        """
        num_cities = int(X.shape[0])
        n_qubits = num_cities ** 2
        # make sure that the sums of x_{i,p} over p do not exceed 1
        for i in range(num_cities):
            if np.sum(X[i, :]) > 1:
                active_columns = [p for p in range(num_cities) if X[i, p] == 1]
                chosen_column = random.choice(active_columns)
                X[i, :] = np.zeros(num_cities)
                X[i, chosen_column] = 1
        # make sure that the sums of x_{i,p} over i do not exceed 1
        for p in range(num_cities):
            if np.sum(X[:, p]) > 1:
                active_rows = [i for i in range(num_cities) if X[i, p] == 1]
                chosen_row = random.choice(active_rows)
                X[:, p] = np.zeros(num_cities)
                X[chosen_row, p] = 1
        # make sure the sum of each row is not zero
        # for each row i
        for i in range(num_cities):
            # if the row is empty
            if np.sum(X[i, :]) == 0:
                free_columns = [p for p in range(num_cities) if np.sum(X[:, p]) == 0]
                chosen_column = random.choice(free_columns)
                X[i, chosen_column] = 1
        # make sure the sum of each column is not zero
        # for each column p
        for p in range(num_cities):
            # if the column is empty
            if np.sum(X[:, p]) == 0:
                free_rows = [i for i in range(num_cities) if np.sum(X[i, :]) == 0]
                chosen_row = random.choice(free_rows)
                X[chosen_row, p] = 1
        # convert the X array back to a string
        string = ''
        for q in range(n_qubits):
            # city i and tour step p encoded by qubit q
            i, p = TSP_QAOA_SOLVER.get_city_and_tour_from_qubit(q, num_cities)
            string += str(int(X[i, p]))
        # return a string that encodes a Hamiltonian path
        return string

    @staticmethod
    def path_from_string(string):
        """
        Converts the bitstring read out of the QAOA circuit into a path over the cities. The validity of the path
        represented by the string is checked and error correction is used whenever needed.
        :param string: string with the values of the qubits
        :return: path as list of city indices
        """
        n_qubits  = len(string)
        num_cities = int(math.sqrt(n_qubits))
        # Check the validity of the path
        validity, X = TSP_QAOA_SOLVER.is_valid_path(string)
        # If the string represents an invalid path
        if validity==False:
            # correct the string
            string = TSP_QAOA_SOLVER.correct_path(X)
        # Builds a path dictionary from the string
        path_dict = {}
        for q in range(n_qubits):
            qubit_value = string[q]
            if qubit_value == '1':
                # we visit city u at tour step i
                u, i = TSP_QAOA_SOLVER.get_city_and_tour_from_qubit(q, num_cities)
                path_dict.update({i: u})
        # Converts the path dictionary into a list of cities
        path = [path_dict[i] for i in range(num_cities)]
        return path

    def compute_path_length(self, path):
        """
        Computes the length of a path
        :param path: list of cities
        :return: total distance
        """
        length = 0
        for i, j in zip(path[:-1], path[1:]):
            length += self.distance_matrix[i, j]
        return length

    def compute_expectation(self, counts):
        """
        Computes expectation of distance based on measurement results

        Args:
            counts: dict
                    key as bitstring, val as count

            AdjMatrix: Adjacency matrix as numpy array

        Returns:
            avg: float
                 expectation value
        """
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            path = self.path_from_string(bitstring)
            path_length = self.compute_path_length(path + [path[0]])
            avg += path_length * count
            sum_count += count
        # return the expectation of the path length
        return avg / sum_count

    def execute_circ(self, qaoa_circuit, theta):
        """
        Executes the QAOA circuit on the qasm simulator or "optimizer backend" and returns the expected value of the
        length of the Hamiltonian tours/paths.
        :param qaoa_circuit:
        :param theta: QAOA circuit parameters of length 2p, starting with p beta values and followed by p gamma values
        :return: expected value of the length of the paths returned by the QAOA circuit with the given parameters theta
        """
        # assigns the parameters values to the QAOA circuit
        bound_qc = qaoa_circuit.bind_parameters(theta)
        # run the circuit with the bound values
        results_dict = self.optimizer_backend.run(bound_qc, seed_simulator=10, nshots=self.num_shots).result().get_counts()
        # compute the expected length of the paths using the results collected
        expectation = self.compute_expectation(results_dict)
        #print("Circuit ran! Expectation: ", expectation, end="\r")
        # return the expected value
        return expectation

    def show_circuit(self, qc):
        depth = qc.decompose().depth()
        count_ops_dict = qc.decompose().count_ops()
        if 'cx' in count_ops_dict.keys():
            cnot_gates = count_ops_dict["cx"]
        else:
            cnot_gates = 0
        print('QAOA circuit has depth', depth, 'and uses', cnot_gates, 'CNOT gate(s)')
        #plot = qc.decompose().draw()
        #print(plot)

    def get_optimized_qaoa_circuit(self):
        """
        Builds an optimized QAOA circuit.
        :return: quantum circuit bound to optimized parameters.
        """
        # Build the QAOA circuit
        qaoa_circuit, beta, gamma = self.create_qaoa_circuit()
        # Show the circuit
        self.show_circuit(qaoa_circuit)
        # if the circuit has parametersnumber of parameters of the circuit
        if qaoa_circuit.num_parameters > 0:
            # Define the expectation operator given circuit parameter values
            expectation = lambda theta: self.execute_circ(qaoa_circuit, theta)
            # Define the initial point
            initial_point = np.random.random(qaoa_circuit.num_parameters)
            # Minimize the expectation with respect to the circuit parameters
            if self.classical_optimizer != 'SPSA':
                optimization_result = minimize(expectation, initial_point, method=self.classical_optimizer)
                theta_star = optimization_result.x
            else:
                spsa = SPSA(maxiter=100)
                result = spsa.optimize(qaoa_circuit.num_parameters, expectation, initial_point=initial_point)
                theta_star = list(result[0])
            # assigns the optimal parameters values to the QAOA circuit
            optimized_qc = qaoa_circuit.bind_parameters(theta_star)
            print('    optimized QAOA circuit parameters: ', theta_star)
        else:
            optimized_qc = qaoa_circuit
        # return optimized circuit
        return optimized_qc

    @staticmethod
    def filter_highest_counts(results_dict, max_rank=1):
        """
        Filters results with highest counts
        :param results_dict: dictionary of results from shots
        :param max_rank: maximum rank of results kept
        :return: filtered dictionary
        """
        filtered_results_dict = {}
        counts = list(results_dict.values())
        counts = list(np.unique(np.array(counts)))
        counts.sort(reverse=True)
        thresholds = counts[:max_rank]
        for threshold in thresholds:
            for key in results_dict.keys():
                if results_dict[key] == threshold:
                    filtered_results_dict.update({key: results_dict[key]})
        return filtered_results_dict

    @staticmethod
    def binary_to_integer(s):
        """
        Convert binary string into integer
        :param s: bit string
        :return: positive or negative integer
        """
        N = len(s)
        n = 0
        for i in range(N):
            n += int(s[N - 1 - i]) * (2 ** i)
        return n

    @staticmethod
    def show_histogram(all_results, results_dict, problem_name):
        ticks = []
        labels = []
        counts = []
        ticks_peaks = []
        labels_peaks = []
        counts_peaks = []
        for read in all_results.keys():
            i = TSP_QAOA_SOLVER.binary_to_integer(read)
            count = all_results[read]
            ticks.append(i)
            counts.append(count)
            if read in results_dict.keys():
                labels.append(read)
                labels_peaks.append(read)
                ticks_peaks.append(i)
                counts_peaks.append(count)
            else:
                labels.append('')
        f, ax = plt.subplots(1)
        ax.bar(ticks, counts, color='b')
        ax.bar(ticks_peaks, counts_peaks, color='r')
        ax.set_title(problem_name)
        ax.set_xticks(ticks_peaks, labels_peaks)
        ax.set_xlim(left=min(ticks), right=max(ticks))
        ax.tick_params(axis='x', labelrotation=90)
        file_name = 'Histogram ' + problem_name + '.png'
        f.savefig(file_name)
        plt.clf()

    def solve(self):
        """
        Solve the TSP problem using the specified algorithm for QAOA .
        :return: shortest_path, minimum_length, execution time
        """
        print('Solving TSP with ' + str(self.num_cities) + ' cities using ' + self.algorithm + ', p=' + str(self.p) +
              ' repetitions and the ' + self.classical_optimizer + ' optimizer:')
        print(self.distance_matrix)
        start_time = time.perf_counter()
        # get the optimized QAOA circuit using the qasm simulator (optimizer backend)
        optimized_qaoa_circuit = self.get_optimized_qaoa_circuit()
        # plot circuit
        #try:
        #    optimized_qaoa_circuit.draw(output='mpl', filename='QAOA circuit TSP#' + self.problem_name + '.png')
        #except:
        #    print('This circuit too large to be printed!')
        # run the optimized QAOA circuit using the specified solver backend
        all_results_dict = execute(optimized_qaoa_circuit, self.solver_backend, shots=self.num_shots).result().get_counts(0)
        # keeps the results with lowest rank
        results_dict = self.filter_highest_counts(all_results_dict, max_rank=1)
        # show the histogram
        #now = datetime.now()
        #dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        #self.show_histogram(all_results_dict, results_dict, 'TSP ' + dt_string)
        # keep the path with minimum distance from the results
        shortest_path = []
        minimum_length = np.inf
        for path_string in results_dict.keys():
            path = self.path_from_string(path_string)
            path_length = self.compute_path_length(path)
            if path_length < minimum_length:
                minimum_length = path_length
                shortest_path = path
        # return the shortest path and its length
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print('    shortest path found:', shortest_path)
        print('    shortest length found:', minimum_length * self.scaling_factor)
        print('    execution time:', execution_time)
        return shortest_path, minimum_length * self.scaling_factor, execution_time

    def solve_classically_by_brute_force(self):
        """
        Enumerates all paths or routes and finds the shortest one.
        :param distance_matrix: square distance matrix as numpy array
        :return: shortest_path, minimum_length, execution_time, maximum_length
        """
        print('Solving TSP classically using brute force:')
        start_time = time.perf_counter()
        paths = list(itertools.permutations([i for i in range(self.num_cities)]))
        minimum_length = np.inf
        maximum_length = 0
        for path in paths:
            length = sum([self.distance_matrix[path[i], path[i + 1]] for i in range(self.num_cities - 1)])
            if self.hamiltonian_path == False:
                length += self.distance_matrix[path[self.num_cities - 1], path[0]]
            if length < minimum_length:
                minimum_length = length
                shortest_path = list(path)
            if length > maximum_length:
                maximum_length = length
        # return the shortest path and its length
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print('    shortest path:', shortest_path)
        print('    shortest length:', minimum_length * self.scaling_factor)
        print('    execution time:', execution_time)
        return shortest_path, minimum_length * self.scaling_factor, execution_time, maximum_length


