from pyqubo import Binary, Array
import neal
import greedy
import numpy as np
import random


class QUBO_SA_SOLVER():
    """
    Class for solving a QUBO of the form min x^T Q x using Simulated Annealing.
    """

    def __init__(self, Q):
        """
        Constructor.
        :param Q: n x n square matrix as numpy array
        """
        # number of binary variables
        self.n = Q.shape[0]
        # array of binary variables
        x = Array.create('x', shape=(self.n), vartype='BINARY')
        # Hamiltonian
        H = 0
        for i in range(self.n):
            for j in range(self.n):
                H += Q[i, j] * x[i] * x[j]
        self.model = H.compile()
        print(self.model.to_qubo())
        self.bqm = self.model.to_bqm()
        self.sampler = neal.SimulatedAnnealingSampler()
        #self.sampler = greedy.SteepestDescentSampler()

    def solve(self, num_reads=1000):
        sampleset = self.sampler.sample(self.bqm, num_reads=num_reads)
        decoded_samples = self.model.decode_sampleset(sampleset)
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        print(best_sample.sample)
        return best_sample

