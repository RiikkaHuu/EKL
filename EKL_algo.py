import numpy as np
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent
from pymanopt import Problem


"""
    Copyright (C) 2019  Riikka Huusari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

This file contains the EKL algorithm(s) and the required helper functions. To use EKL, you should do something like 
the following:  

  from EKL_algo import EKL
  ekl = EKL(features, labels, l, r, g)  # l: lambda, r: rank, g: gamma  (as in the EKL paper)
  ekl.solve()  # manifold optimisation can be slow... 
  predictions = ekl.predict(test_features)
  ptr_predictions = ekl.ptr_predict(test_features)  # the partial trace formulation, see paper

If you want to predict with new lambda values but keeping the entangled kernel (gamma parameter) the same, you can 
do it by

  predictions = ekl.predict(test_features, l=new_lambda)
  ptr_predictions = ekl.ptr_predict(test_features, l=new_lambda)

The code is dependent on the Pymanopt package, see https://www.pymanopt.org/
"""


# ================================================= helper functions ===================================================

def centering_mtrx(n):

    return np.eye(n) - (1 / n) * np.ones((n, n))


def partial_trace(matrix, n, p):

    newmat = np.zeros((n, n))

    for ii in range(n):
        for jj in range(n):
            newmat[ii, jj] = np.trace(matrix[ii*p:(ii+1)*p, jj*p:(jj+1)*p])

    return newmat

# ======================================= alignments (helpers for EKL class) ===========================================

# the alignments G_yyT and ptrG_YYT are the two parts of the kernel optimization problem introduced in the paper
# with python


def alignment_G_yyT(phi, Y, U):

    # about naming: Q is here U

    [p, n] = Y.shape
    C = centering_mtrx(n*p)

    #
    CY = np.reshape(np.dot(C, np.reshape(Y, [-1, 1], order='F')), [p, n], order='F')

    vecYphiT = np.reshape(np.dot(CY, np.transpose(phi)), [-1, 1], order='F')
    UTvecYphiT = np.dot(np.transpose(U), vecYphiT)

    upstairs = np.dot(np.transpose(UTvecYphiT), UTvecYphiT)

    #
    UTphiIC = np.dot(np.dot(np.transpose(U), np.kron(phi, np.eye(p))), C)
    downstairs = np.trace(np.dot(np.dot(UTphiIC, np.transpose(UTphiIC)), np.dot(UTphiIC, np.transpose(UTphiIC))))

    return upstairs / np.sqrt(downstairs)


def alignment_ptrG_YYT(phi, Y, U):

    # about naming: Q is here U

    # assume phi is m*n, then kernel is (phiT I)D(phi I)
    [p, n] = Y.shape
    [m, _] = phi.shape
    # [mp, _] = U.shape

    D = np.dot(U, np.transpose(U))

    # calculate  partial trace of D
    ptD = np.zeros((m, m))

    for n1 in range(m):
        for n2 in range(m):

            # # version with autograd numpy
            # tmpmat = np.zeros((m, m))
            # tmpmat[n1, n2] = 1
            #
            # ptD = ptD + np.trace(D[n1 * p:(n1 + 1) * p, n2 * p:(n2 + 1) * p]) * tmpmat

            ptD[n1, n2] = np.trace(D[n1 * p:(n1 + 1) * p, n2 * p:(n2 + 1) * p])  # autograd numpy doesn't accept this

    C = centering_mtrx(n)

    YCphiT = np.dot(np.dot(Y, C), np.transpose(phi))
    upstairs = np.trace(np.dot(YCphiT, np.dot(ptD, np.transpose(YCphiT))))

    PtrDphiCphiT = np.dot(ptD, np.dot(phi, np.dot(C, np.transpose(phi))))

    downstairs = np.trace(np.dot(PtrDphiCphiT, PtrDphiCphiT))
    alignment = upstairs / np.sqrt(downstairs)

    return alignment


# =================================== alignment derivatives (helpers for EKL class)=====================================


def derivative_alignment_ptrG_YYT(phi, Y, U):

    # about naming: Q is here U

    # assume phi is m*n, then kernel is (phiT I)D(phi I)
    [p, n] = Y.shape
    [m, _] = phi.shape

    D = np.dot(U, np.transpose(U))

    ptD = np.zeros((m, m))

    for n1 in range(m):
        for n2 in range(m):

            # # version with autograd numpy
            # tmpmat = np.zeros((m, m))
            # tmpmat[n1, n2] = 1
            #
            # ptD = ptD + np.trace(D[n1 * p:(n1 + 1) * p, n2 * p:(n2 + 1) * p]) * tmpmat

            ptD[n1, n2] = np.trace(D[n1 * p:(n1 + 1) * p, n2 * p:(n2 + 1) * p])  # autograd numpy doesn't accept this

    C = centering_mtrx(n)

    YCphiT = np.dot(np.dot(Y, C), np.transpose(phi))
    upstairstrace = np.trace(np.dot(np.transpose(U), np.dot(np.kron(np.dot(np.transpose(YCphiT), YCphiT),
                                                                         np.eye(p)), U)))

    PtrDphiCphiT = np.dot(ptD, np.dot(phi, np.dot(C, np.transpose(phi))))

    downstairstrace = np.trace(np.dot(PtrDphiCphiT, PtrDphiCphiT))

    # derivatives of the parts
    dupstairs = 2*np.dot(np.kron(np.dot(np.transpose(YCphiT), YCphiT), np.eye(p)), U)
    phiCphiT = np.dot(phi, np.dot(C, np.transpose(phi)))
    ddownstairstrace = 4 * np.dot(np.kron(np.dot(phiCphiT, np.dot(ptD, phiCphiT)), np.eye(p)), U)
    ddownstairs = ddownstairstrace/(2*np.sqrt(downstairstrace))

    # combine to the final derivative with the quotient rule
    derivative = (dupstairs*np.sqrt(downstairstrace)-upstairstrace*ddownstairs)/downstairstrace

    return derivative


def derivative_alignment_G_yyT(phi, Y, U):

    # about naming: Q is here U

    [p, n] = Y.shape
    C = centering_mtrx(n*p)

    CY = np.reshape(np.dot(C, np.reshape(Y, [-1, 1], order='F')), [p, n], order='F')

    vecYphiT = np.reshape(np.dot(CY, np.transpose(phi)), [-1, 1], order='F')
    UTvecYphiT = np.dot(np.transpose(U), vecYphiT)

    upstairstrace = np.trace(np.dot(np.transpose(UTvecYphiT), UTvecYphiT))  # this should already be a scalar

    # this downstairs version cannot be calculated as efficiently
    UTphiIC = np.dot(np.dot(np.transpose(U), np.kron(phi, np.eye(p))), C)
    downstairstrace = np.trace(np.dot(np.dot(UTphiIC, np.transpose(UTphiIC)), np.dot(UTphiIC, np.transpose(UTphiIC))))

    # derivatives of the parts
    dupstairs = 2*np.dot(vecYphiT, np.dot(np.transpose(vecYphiT), U))
    ddownstairstrace = 4*np.dot(np.dot(np.kron(phi, np.eye(p)), np.transpose(UTphiIC)),
                                 np.dot(UTphiIC, np.transpose(UTphiIC)))
    ddownstairs = ddownstairstrace / (2 * np.sqrt(downstairstrace))

    # combine to the final derivative with the quotient rule
    derivative = (dupstairs * np.sqrt(downstairstrace) - upstairstrace * ddownstairs) / downstairstrace

    return derivative


# ======================================================================================================================
# ====================================================== EKL ===========================================================
# ======================================================================================================================


class EKL:

    def __init__(self, phi, Y, lmbda, rank, gamma):

        """

        :param phi: n*m matrix containing the features given to EKL algorithm
        :param Y: n*p matrix containing the labels of the prediction problem
        :param lmbda: regularization parameter for learning the vector-valued function for predicting
        :param rank: rank of the entangled kernel to be learned, or number of columns in matrix Q
        :param gamma: regularization parameter for learning the entangled kernel, between 0 and 1 (inclusive)
        """

        self.phi = np.transpose(phi)
        self.Y = np.transpose(Y)
        self.lmbda = lmbda
        self.gamma = gamma
        self.rank = rank

        # check the input sizes:
        [m, n1] = self.phi.shape
        [p, n2] = self.Y.shape

        if n1 != n2:
            sizestr = str(n1) + "*" + str(m) + " and " + str(n2) + "*" + str(p)
            exit("sizes of Phi and Y should be n*m and n*p, you gave them as "+sizestr)

        self.n = phi.shape[0]
        self.m = phi.shape[1]
        self.p = Y.shape[1]

        self.c = None
        self.D = None
        self.Q = None

    def solve(self):

        """
        Solves the EKL optimization problem; first performs the Q update for learning the kernel, then c update
        that learns the predictive function (this is done for both operator-valued and scalar-valued cases).
        """

        # ========= STEP 1 - D =========

        # gradient ascent over Q, D = QQ^T

        self.Q = self._Q_update()
        self.D = np.dot(self.Q, np.transpose(self.Q))

        # ========= STEP 2 - c =========

        # the basic solution with Woodbury formula

        self._c_update()
        self._ptr_c_update()

    def predict(self, test_phi, l=-1):

        """
        Predict labels of new samples based on the features given in test_phi
        :param test_phi: t*m matrix containing the features for the test data
        :param l: optional new value for regularization parameter lambda, if changed the predictive function is updated
        accordingly (will change the class parameter)
        :return: predictions, a t*p matrix
        """

        if l > 0:  # if given new lambda update the predictive function first
            self.lmbda = l
            self._c_update()

        test_phi = np.transpose(test_phi)

        # need to vectorize things, so this looks a bit messy
        vecCK = np.reshape(np.dot(np.reshape(self.c, (self.p, self.n), order='F'), np.transpose(self.phi)),
                           (-1, 1), order='F')
        tmpmat = np.reshape(np.dot(self.D, vecCK), (self.p, self.m), order='F')

        pred = np.dot(tmpmat, test_phi)

        return np.transpose(pred)

    def ptr_predict(self, test_phi, l=-1):

        """
        Predict labels of new samples based on the features given in test_phi using the partial trace formulation
        :param test_phi: t*m matrix containing the features for the test data
        :param l: optional new value for regularization parameter lambda, if changed the predictive function is updated
        accordingly (will not change the class parameter)
        :return: predictions, a t*p matrix
        """

        if l > 0:  # update c first in this case
            self._ptr_c_update(l=l)

        ptrD = partial_trace(self.D, self.m, self.p)

        preds = np.dot(test_phi, np.dot(ptrD, np.dot(self.phi, self.ptr_c)))

        return preds

    def _c_update(self):

        # The solution with Woodbury formula! See matrix cookbook
        # When we have Q we only need one inverse of size rank*rank (at most mp*mp), with D we need 2 with mp*mp.

        inverse = np.linalg.pinv(np.eye(self.rank) +
                                 (1/self.lmbda) * np.dot(np.transpose(self.Q),
                                                         np.dot(np.kron(np.dot(self.phi, np.transpose(self.phi)),
                                                                        np.eye(self.p)), self.Q)))

        vecYK = np.reshape(np.dot(self.Y, np.transpose(self.phi)), (-1, 1), order='F')

        tmpvec = np.dot(self.Q, np.dot(inverse, np.dot(np.transpose(self.Q), vecYK)))

        part2 = np.reshape(np.dot(np.reshape(tmpvec, [self.p, self.m], order='F'), self.phi),
                           [self.n * self.p, 1], order='F')

        self.c = (1 / self.lmbda) * np.reshape(self.Y, (-1, 1), order="F") - \
                 (1 / self.lmbda ** 2) * part2

    def _ptr_c_update(self, l=-1):

        if l<0:
            lmbda = l
        else:
            lmbda = self.lmbda

        ptrD = partial_trace(self.D, self.m, self.p)

        c = (1/lmbda)*np.transpose(self.Y)
        inverse = np.linalg.pinv(np.linalg.pinv(ptrD) + (1/lmbda) * np.dot(self.phi, np.transpose(self.phi)))
        c = c + (1/lmbda**2)*np.dot(np.transpose(self.phi),
                                    np.dot(inverse, np.dot(self.phi, np.transpose(self.Y))))

        self.ptr_c = c

    def _Q_update(self):

        # updates the Q parameter by constructing the loss and its gradient, and calls the function solving manifold
        # optimization
        # note: minus in front because pymanopt does gradient descent, EKL wants ascent

        def Qloss(U):

            return -((1-self.gamma) * alignment_ptrG_YYT(self.phi, self.Y, U)
                     + self.gamma * alignment_G_yyT(self.phi, self.Y, U))

        def Qgrad(U):

            return -((1-self.gamma) * derivative_alignment_ptrG_YYT(self.phi, self.Y, U) +
                     self.gamma * derivative_alignment_G_yyT(self.phi, self.Y, U))

        return self._Q_solver(Qloss, Qgrad)

    def _Q_solver(self, Qloss, Qgrad, printprogress=0):

        Qinit = np.random.rand(self.m * self.p, self.rank)
        # need to normalize Qinit to sphere manifold
        Qinit = Qinit / np.linalg.norm(Qinit, ord='fro')

        manifold = Sphere(self.p * self.m, self.rank)

        problem = Problem(manifold=manifold, cost=Qloss, verbosity=printprogress, egrad=Qgrad)

        problemsolver = SteepestDescent(maxiter=200)

        Q = problemsolver.solve(problem, x=Qinit)

        return Q



