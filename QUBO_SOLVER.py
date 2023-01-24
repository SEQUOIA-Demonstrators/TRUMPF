from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import QAOA, VQE
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
#from qiskit_methods.backends import *
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, SciPyOptimizer
#from qiskit.circuit.library import EfficientSU2



class QUBO_SOLVER():

    def __init__(self, binary_variables, linear_coefficients, quadratic_coefficients, backend):
        """

        :param binary_variables: list of strings with the names of the binary variables
        :param linear_coefficients: list of value of the coefficients for each binary variable
        :param quadratic_coefficients: dictionary with key: value pairs of the form (var1, var2): coefficient
        :param backend: quantum backend on which to solve the QUBO
        """
        self.num_qubits_needed = len(binary_variables)
        self.backend = backend
        # create a QUBO in qiskit
        qubo = QuadraticProgram()
        for b in binary_variables:
            qubo.binary_var(b)
        qubo.minimize(linear=linear_coefficients, quadratic=quadratic_coefficients)
        print(qubo.export_as_lp_string())
        self.qubo = qubo

    def resolve(self, algorithm='QAOA'):
        # Convert QUBO to Ising model
        operator, offset = self.qubo.to_ising()
        # Set random seed
        algorithm_globals.random_seed = 42
        # Set a classical optimizer
        #optimizer = COBYLA()
        #optimizer = L_BFGS_B()
        #optimizer = SLSQP()
        optimizer = SPSA()
        # Create quantum instance
        quantum_instance = QuantumInstance(self.backend,
                                           seed_simulator=algorithm_globals.random_seed,
                                           seed_transpiler=algorithm_globals.random_seed)
        # https://qiskit.org/documentation/stable/0.25/stubs/qiskit.algorithms.QAOA.html
        if algorithm == 'QAOA':
            mesurement = QAOA(optimizer=optimizer, reps=1, quantum_instance=quantum_instance)
        # https://qiskit.org/documentation/stubs/qiskit.algorithms.VQE.html
        elif algorithm == 'VQE':
            mesurement = VQE(optimizer=optimizer, quantum_instance=quantum_instance)
        else:
            print('Warning: algorithm must be either QAOA or VQE')
        # Minimum eigen optimization problem
        meop = MinimumEigenOptimizer(mesurement)
        # Solve the minimum eigen optimization problem
        result = meop.solve(self.qubo)
        # Get sampled solutions
        samples = result.samples
        # Get the samples vectors
        sampled_vectors = [sample.x for sample in samples]
        # Names of the QUBO variables sampled
        variable_names = result.variable_names
        # Collect solutions in the form of a list of dictionaries
        solutions = []
        for v in sampled_vectors:
            # build a dictionary that maps the variable names to the components of vector v
            solution_dictionary = {}
            for i in range(len(v)):
                solution_dictionary.update({variable_names[i]: v[i]})
            # append the dictionary to the list of solutions
            solutions.append(solution_dictionary)
        # Return the solutions of the QUBO
        return solutions



