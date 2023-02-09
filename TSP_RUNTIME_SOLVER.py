import math
import numpy as np
from qiskit import BasicAer
from qiskit.utils import algorithm_globals, QuantumInstance
#from qiskit.algorithms import QAOA
from qiskit.algorithms import VQE
#from qiskit_optimization.runtime import QAOAClient
#from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import EfficientSU2
import time


def main(backend, user_messenger, problem_instances):
    """
    Main program for our TSP RUNTIME SOLVER which solves a batch of Traveling Salesman Problems.

    :param backend: Qiskit backend instance (ProgramBackend)
    :param user_messenger: Used to communicate with the program user (UserMessenger)
    :param problem_instances: a dictionary of problem instances, where keys are the TSP instance name as string and
                              values are distance matrices as lists of lists.
    :return: dictionary of solutions (Hamiltonian path as list of vertices) for each instance
    """

    def callback(dict):
        user_messenger.publish(dict)

    intermediate_info = {
        'nfev': [],
        'parameters': [],
        'energy': [],
        'stddev': []
    }

    def vqe_callback(nfev, parameters, energy, stddev):
        intermediate_info['nfev'].append(nfev)
        intermediate_info['parameters'].append(parameters)
        intermediate_info['energy'].append(energy)
        intermediate_info['stddev'].append(stddev)

    def build_TSP_objective_function(W):
        N = W.shape[0]
        # maximum of absolute distances
        Wmax = np.abs(W).max()
        # Penalty coefficient
        A = 2.0 * math.ceil(Wmax)
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

    def is_Hamiltonian_path(result):
        num_vertices = int(math.sqrt(len(list(result.keys()))))
        x_matrix = np.zeros((num_vertices, num_vertices))
        for binary_variable in result.keys():
            strings = binary_variable.split('_')
            i = int(strings[1])
            p = int(strings[2])
            x_matrix[i - 1, p - 1] = result[binary_variable]
        for p in range(num_vertices):
            if np.sum(x_matrix[:, p]) != 1.0:
                return False
        for i in range(num_vertices):
            if np.sum(x_matrix[i, :]) != 1.0:
                return False
        return True

    def decode_TSP_solution(results_dictionary):
        num_cities = int(math.sqrt(len(results_dictionary.keys())))
        hamiltonian_path = np.zeros(num_cities)
        for var in results_dictionary.keys():
            parsed_variable_name = var.split('_')
            i = int(parsed_variable_name[1])
            p = int(parsed_variable_name[2])
            value = results_dictionary[var]
            if int(value) == 1:
                hamiltonian_path[p-1] = i
        return hamiltonian_path

    # initialize dictionary of solutions
    solutions_dict = {}

    # for each problem instance
    for instance_name in problem_instances.keys():
        print('Solving TSP: ' + instance_name)
        # initialize compute time statistics
        num_trials = 0
        start_time = time.perf_counter()
        # get the distance matrix D
        D = problem_instances[instance_name]
        # Convert distance matrix to a numpy array
        D = np.asarray(D)
        # Determine binary variable names and linear and quadratic coefficients of the QUBO used for solving the TSP
        binary_variables, linear_coefficients, quadratic_coefficients = build_TSP_objective_function(D)
        # Number of qubits needed
        num_qubits = len(binary_variables)
        # QUBO for the TSP
        qubo = QuadraticProgram()
        for b in binary_variables:
            qubo.binary_var(b)
        qubo.minimize(linear=linear_coefficients, quadratic=quadratic_coefficients)
        print(qubo.export_as_lp_string())
        # Convert QUBO to Ising and get a qubit operator
        hamiltonian, offset = qubo.to_ising()
        print("Offset:", offset)
        print("Operator:")
        print(str(hamiltonian))
        # Set random seed
        algorithm_globals.random_seed = 10598
        # Create quantum instance
        quantum_instance = QuantumInstance(backend, seed_simulator=algorithm_globals.random_seed, seed_transpiler=algorithm_globals.random_seed)

        # Use QAOA to find the minimum eigenvalue of the Ising operator
        #p = 1 # number of repetitions of the alternating ansatz
        #spsa_optimizer = SPSA(maxiter=100)
        #qaoa_mes = QAOA(optimizer=spsa_optimizer, reps=p, quantum_instance=quantum_instance)
        #qaoa_mes = QAOAClient(provider=provider, backend=backend, optimizer=spsa_optimizer, reps=p, alpha=0.1)
        #qaoa = MinimumEigenOptimizer(qaoa_mes)
        #result = qaoa.solve(qubo)

        # Use VQE
        p = 1 # number of repetitions of the ansatz structure
        optimizer = SPSA(maxiter=100)
        # the rotation gates are chosen randomly, so we set a seed for reproducibility
        ansatz = EfficientSU2(num_qubits, reps=p, entanglement='linear', insert_barriers=True)
        vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance, callback=vqe_callback)
        # As long as we don't have a valid Hamiltonian path
        is_valid = False
        while is_valid == False:
            num_trials += 1
            print('============================')
            print('Trial #' + str(num_trials))
            result = vqe.compute_minimum_eigenvalue(hamiltonian)
            for eigenvector in result.eigenstate.keys():
                # convert the bit string to a dictionary that assigns to each binary variable a value
                result_dictionary = {}
                for pos in range(len(eigenvector)):
                    var = binary_variables[pos]
                    value = eigenvector[pos]
                    result_dictionary.update({var: value})
                # Check validity of the solution
                is_valid = is_Hamiltonian_path(result_dictionary)
                print('    solution is a Hamiltonian path: ', is_valid)
                # stop when we have a Hamiltonian path
                if is_valid:
                    break
        # Decodes the solution of the QUBO into a Hamiltonian path (list of vertices)
        hamiltonian_path = decode_TSP_solution(result_dictionary).astype(int).tolist()
        # stop timer
        end_time = time.perf_counter()
        # Store the solution Hamiltonian_path for this problem instance
        solutions_dict.update({instance_name: hamiltonian_path})
        # Store compute time statistics
        solutions_dict.update({instance_name + '_num_trials': num_trials, instance_name + '_compute_time': end_time - start_time})
        # Show interim solutions
        callback(solutions_dict)
    # Return the dictionary of valid solutions
    return solutions_dict

# Metadata
meta = {
  "name": "TSP-RUNTIME_SOLVER",
  "description": "Solves a batch of Traveling Salesman Problems and ensures that the solutions are Hamiltonian paths.",
  "max_execution_time": 100000,
  "spec": {}
}

meta["spec"]["parameters"] = {
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "properties": {
    "problem_instances": {
      "description": "",
      "type": "a dictionary of problem instances, where keys are the instance name as string and values are a "
              "distance matrices as lists of lists."
    },
  },
  "required": [
    "problem_instances"
  ]
}

meta["spec"]["return_values"] = {
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "description": "dictionary of Hamiltonian paths for each TSP instance",
  "type": "dictionary"
}

meta["spec"]["interim_results"] = {
  "$schema": "https://json-schema.org/draft/2019-09/schema",
  "description": "dictionary of solutions found so far",
  "type": "dictionary"
}