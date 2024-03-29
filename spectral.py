import numpy as np
import numpy.polynomial as P
import math
from scipy.special import chebyt
from scipy.integrate import quad
import scipy.linalg

class util:
    def is_in_range(x, x1, x2):
        return x1 <= x and x <= x2
    def c_n(n):
        return 2 if n == 0 else 1
    def delta(n, m):
        return 1 if n == m else 0

class wavefunc:
    def __init__(self, N = 10, V = [0.0, 0.5], Q1 = -1.0, Q2 = 1.0, Nv = None, Vargs = None):
        self.N = N
        self.M = N + 1

        self.hbar = 1
        self.m = 1
        self.a0 = 1
        self.E0 = self.m * self.a0 **2 / self.hbar ** 2

        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = 0.5 * (Q1 + Q2)
        self.Q0 = 0.5 * (Q2 - Q1)
        
        if Nv != None:
            if Vargs != None:
                V2 = lambda x: V(x, Vargs)
            else:
                V2 = V
            self.U = self.function_potential_coefs(V2, Nv)
        else:
            self.U = P.Polynomial(self.calculate_potential_coef(V / self.E0)).convert(kind=P.Chebyshev)
        self.Nu = self.U.degree()
        
        # Av=eBv => generalized Eigenvalue problem
        A = [[self.A_mel(j, k) for k in range(N + 1)] for j in range(N + 1)]
        B = [[self.B_mel(j, k) for k in range(N + 1)] for j in range(N + 1)]
        eigenvals, eigenvecs = scipy.linalg.eig(A, B, right = True)
        self.sort_eigenvals(eigenvals, eigenvecs)


    def function_potential_coefs(self, V, Nv):
        integrand = lambda x, n: V(self.Q0 * self.a0 * x + self.a0 * self.Q3)\
              * chebyt(n)(x) / np.sqrt(1-x**2)
        U = [None] * Nv
        c_n = 2
        for n in range(Nv):
            vnl = quad(integrand,-1, 1, args=n)
            U[n] = vnl[0] / self.E0 * 2/np.pi / c_n
            c_n = 1
        return P.Chebyshev(U)
    
    def A_mel(self, j, k):
        if j == self.N - 1:
            return self.first_bc(k)
        elif j == self.N:
            return self.second_bc(k)
        else:
            return -self.eigenvalue_elements(j, k)
    def B_mel(self, j, k):
        if j == k and j <= self.N - 2:
            return -1.0
        else:
            return 0.0
    # Boundary condition \psi(x=-1) = 0
    def first_bc(self, n):
        if util.is_in_range(n, 0, self.N):
            return -1.0 if n % 2 == 1 else 1.0
        else: return 0.0
    # Boundary condition \psi(x=+1) = 0
    def second_bc(self, n):
        if util.is_in_range(n, 0, self.N):
            return 1.0
        else: return 0.0

    def eigenvalue_elements(self, k : int, n : int):
        return -1/(2 * self.Q0**2) * self.E_squared_elements(k, n) + 0.5 *  self.A_elements(k, n)
    
    def dimlessHamiltonian(self):
        return np.array([[self.eigenvalue_elements(k, n) for n in range(self.N + 1)] for k in range(self.N + 1)])
    
    def sort_eigenvals(self, vals, vecs):
        valid = []
        for i in range(len(vals)):
            val = vals[i]
            #vec = vecs[:,i]
            if val.imag == 0 and val.real >= 0 and val != np.inf:
                valid.append([val.real, i])
        valid.sort(key = lambda x : x[0])
        self.eigenvecs = []
        self.eigenvals = []
        for val, j in valid:
            self.eigenvals.append(val)
            self.eigenvecs.append(vecs[:,j])
    def E_squared_elements(self, n, p):
        #n -= 1
        #p -= 1
        p_prime = p - n
        if util.is_in_range(n, 0, self.N-3) and util.is_in_range(p, n+2, self.N) and p_prime % 2 == 0:
            return p * (p * p - n * n) / util.c_n(n)
        else:
            return 0.0
    def A_elements(self, k, n):
        a = k - n
        b = n - k
        c = n + k
        vals = [a, b, c]
        d = 1/util.c_n(k)
        multiplier = [1,d,d]
        ret = 0.0
        for i in range(3):
            ell = vals[i]
            if util.is_in_range(ell, 0, self.Nu):
                ret += multiplier[i] * self.U.coef[ell]
        return ret
    
    def calculate_potential_coef(self, V: list):
        U = [self.calculate_nth_potential_coef(V, k) for k in range (len(V))]
        return U
    def calculate_nth_potential_coef(self, V, k):
        ret = 0.0
        for n in range(k, len(V)):
            ret += V[n] * self.a0**n * self.Q3 **(n - k) * math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        return self.Q0 ** k * ret 
    
    def __call__(self, n, q, x_notq = False):
        # If we are calling \psi(x)
        if x_notq:
            x = q
        # If we are calling \psi(q)
        else:
            Q = q / self.a0
            x = (Q - self.Q3) / self.Q0
        ret = 0.0
        coef = self.eigenvecs[n]
        for i in range(self.N + 1):
            ret += coef[i] * chebyt(i)(x)
        return ret

class time_wavefunc:
    def __init__(self, soln : wavefunc, n, times):
        self.times = times
        self.m, self.hbar, self.a0 = soln.m, soln.hbar, soln.a0
        self.Q3, self.Q0 = soln.Q3, soln.Q0
        self.tau = soln.m * soln.a0**2 / soln.hbar
        self.M = len(soln.eigenvecs) # M is the total number of eigenvectors

        ''' 
        We now need to Manually create a hamiltonian with the eigenvals and vecs we calculated in soln.
        H = P D P^-1
        Where P is the eigenvector matrix and D is the diagonal matrix of eigenvalues.
        '''
        self.evals = soln.eigenvals
        self.P = [[soln.eigenvecs[j][i] for j in range(self.M)] for i in range(self.M)]
        self.Pinv = scipy.linalg.inv(self.P)
        self.avec0 = soln.eigenvecs[n][:self.M] # reduce size of vector by number of boundary conditions (ie 2)

    def calc_at(self, times):
        avecs = [self.evolutionOperator(time) @ self.avec0 for time in times]
        return avecs

    def evolutionOperator(self, t):
        U = np.zeros((self.M, self.M), dtype = complex)
        for i in range(self.M):
            U[i][i] = np.exp(1j * t/self.tau * self.evals[i])
        return self.P @ U @ self.Pinv
    
    def __call__(self, q, t):
        Q = q / self.a0
        x = (Q - self.Q3) / self.Q0
        ret = 0.0
        U = self.evolutionOperator(t)
        avec_t = U @ self.avec0
        for i in range(len(avec_t)):
            ret += avec_t[i] * chebyt(i)(x)
        return ret


class wigner:
    def __init__(self, Nx, Ny, V  = [0], Q1 = -10, Q2 = 10, P0 = 10, a0 = 1, hbar = 1, E=0.5):
        self.Nx, self.Ny  = Nx, Ny
        self.Ny2 = int(Ny/2) + 1
        self.N1 = (self.Nx + 1) * (self.Ny2)
        self.N2 = (self.Nx + 1) * (self.Ny + 1) # dimension of our contracted matrices and vectors
        
        self.a0 = a0
        self.hbar = hbar
        self.Q1, self.Q2, self.P0 = Q1, Q2, P0
        self.Q0 = 0.5 * (Q2 - Q1)
        self.Q3 = 0.5 * (Q2 + Q1)
        self.m = 1
        self.E = self.m * self.a0**2 / self.hbar * E # dimensionless energy
        
        Vprime = [self.calculate_nth_potential_coef(V, k) for k in range(len(V))]
        U = P.Polynomial(Vprime).convert(kind=P.Chebyshev)
        self.max_order = np.min((U.degree() + 1, Ny+1))
        self.Ucoefs = self.calculate_potential_coefs(V) # ndarray of Ucoefs[n,l] = U^n_l

    def num_odd(n: int):
        return int((n + 1)/2)

    def calculate_potential_coefs(self, V):
        Vprime = [self.calculate_nth_potential_coef(V, k) for k in range(len(V))]
        U = P.Polynomial(Vprime).convert(kind=P.Chebyshev)
        deg = U.degree()
        Ucoefs = np.zeros((deg + 1, deg + 1), dtype = float)
        Ucoefs[0] = U.coef
        for n in range(1,deg + 1):
            Un = []
            for i in range(deg + 1):
                sum = 0.0
                for k in range(deg + 1):
                    sum += self.derivative_mel(n, i, k) * U.coef[k]
                Ucoefs[n,i] = sum
        return Ucoefs
    
    def calculate_nth_potential_coef(self, V, k):
        ret = 0.0
        for n in range(k, len(V)):
            ret += V[n] * self.a0**n * self.Q3 **(n - k) * math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
        return self.Q0 ** k * ret
    
    def compute(self):
        print("Computing gamma matrix elements...")
        self.gamma = np.zeros((self.N2 + 1, self.N1), )
        for a in range(self.N2):
            if a % int(self.N2 / 10) == 0:
                print('.', end = "", flush = True)
            for b in range(self.N1):
                mel = self.construct_gamma3(a,b)
                self.gamma[a,b] = mel
        print("")
        # Boundary condition W(0,0) = 1
        for a in range(self.N1):
            i, j = self.expand_indices(a)
            j = 2 * j
            if i % 2 == 0 and j % 2 == 0:
                self.gamma[self.N2, a] = ((-1)**(i/2 + j/2))
        print("Preforming Least Squares...")
        # change to i == self.N2 - 1 for coolness
        b = [1 if i == self.N2 else 0 for i in range(self.N2 + 1) ]
        self.vec, residuals, rank, sing = scipy.linalg.lstsq(self.gamma, b)
        R = self.gamma @ self.vec - b
        self.resid = R.transpose() @ R

    def compute2(self):
        print("Computing gamma matrix elements...")
        self.gamma = np.zeros((self.N1, self.N1), )
        for a in range(self.N1):
            if a % int(self.N1 / 10) == 0:
                print('.', end = "", flush = True)
            for b in range(self.N1):
                self.gamma[a,b] = self.construct_gamma4(a, b)
        print("")
        for a in range(self.N1):
            i, j = self.expand_indices(a)
            j = 2 * j
            self.gamma[self.N1 - 2, a] = (-1)**i
            self.gamma[self.N1 - 3, a] = 1

            if i % 2 == 0 and j % 2 == 0:
                self.gamma[self.N1- 1, a] = ((-1)**(i/2 + j/2))
            else:
                self.gamma[self.N1 - 1, a] = 0
        print("Preforming Least Squares...")
        C = np.diag([1 if int(a / (self.Nx+ 1))%2 == 0 else 0 for a in range(self.N1)])
        eigs, self.vecs = scipy.linalg.eig(self.gamma, C)
        self.eigs = []
        for eig in eigs:
            if eig != np.inf and eig != np.nan and eig.imag == 0 and eig.real >0:
                self.eigs.append(eig.real)
        self.eigs.sort()
        self.eigs = np.array(self.eigs)

    def construct_gamma3(self, s, t):
        mu, nu = self.expand_indices(s)
        i, j = self.expand_indices(t)
        j = 2*j 

        if nu %2 == 1:
            p1 = -self.P0/ (2 * self.Q0) * self.derivative_mel(1, mu, i) * \
                (util.delta(j + 1, nu) + util.delta(np.abs(j -1),nu))
            p2 = 0
            for n in np.arange(1, self.max_order, 2):
                b_n = (1j/(2 * self.Q0 * self.P0))**n / math.factorial(n)
                for ell in range(self.max_order - n): # <= Nv - n
                    d = util.delta(ell + i, mu)+ util.delta(np.abs(ell - i), mu)
                    if d != 0:
                        p2 += -1j * b_n * self.Ucoefs[n][ell] * self.derivative_mel(n, nu, j) * d
            return (p1 + p2).real
        elif nu %2 == 0:
            total = 0.0
            if mu == i and nu == j:
                total += self.P0**2 /4 - self.E
            elif mu == i:
                deltas = util.delta(j + 2, nu) + util.delta(np.abs(j - 2), nu)
                total += deltas * self.P0**2 / 8
            if nu == j:
                total -= 1/(8* self.Q0**2) * self.derivative_mel(2, mu, i)
            
            for n in np.arange(0, self.max_order, 2):
                mel = 0
                
                mel = self.derivative_mel(n, nu, j)
                
                bn = (1j/(2 * self.Q0 * self.P0))**n / math.factorial(n)
                for ell in range(self.max_order - n):
                    deltas = util.delta(ell + i,mu) + util.delta(np.abs(ell - i), mu)
                    if deltas != 0:
                        total += 0.5 * bn * self.Ucoefs[n][ell] * mel * deltas
            return total.real

    def construct_gamma4(self, s, t):
        mu, nu = self.expand_indices(s)
        i, j = self.expand_indices(t)
        j*= 2
        if j % 2 != 0:
            return 0

        if nu %2 == 1:
            p1 = -self.P0/ 2 / self.Q0 * self.derivative_mel(1, mu, i) * \
                (util.delta(j + 1, nu) + util.delta(np.abs(j -1),nu))
            p2 = 0
            for n in np.arange(1, self.max_order, 2):
                b_n = (1j/(2 * self.Q0 * self.P0))**n / math.factorial(n)
                for ell in range(self.max_order - n): # <= Nv - n
                    d = util.delta(ell + i, mu)+ util.delta(np.abs(ell - i), mu)
                    if d != 0:
                        p2 += -j * b_n * self.Ucoefs[n][ell] * self.derivative_mel(n, nu, j) * d

            return (p1 + p2).real
        elif nu %2 == 0:
            total = 0.0
            if mu == i and nu == j:
                total += self.P0**2 /4
            elif mu == i:
                deltas = util.delta(j + 2, nu) + util.delta(np.abs(j - 2), nu)
                total += deltas * self.P0**2 / 8
            if nu == j:
                total -= 1/(8* self.Q0**2) * self.derivative_mel(2, mu, i)
            
            for n in np.arange(0, self.max_order, 2):
                mel = 0
                if n == 0:
                    if nu == j:
                        mel = 1
                    else:
                        mel = 0
                else:
                    mel = self.derivative_mel(n, nu, j)
                
                bn = (1j/(2 * self.Q0 * self.P0))**n / math.factorial(n)
                for ell in range(self.max_order - n):
                    deltas = util.delta(ell + i,mu) + util.delta(np.abs(ell - i), mu)
                    if deltas != 0:
                        total += 0.5 * bn * self.Ucoefs[n][ell] * mel * deltas
            
            return total.real

    def derivative_mel(self, n : int, i, j):
        if n == 0:
            return util.delta(i, j)
        if j < n + i or (j + i + n)%2 != 0:
            return 0
        product = 2*j
        sigma = -(n/2 - 1)
        max = n/2 - 1
        while sigma <= max:
            product *= (j*j - (i + 2 * sigma)**2)
            sigma += 1
        try:
            fact = math.factorial(n-1)*2**(n-1)
        except OverflowError as err:
            print(n)
        return product * (0.5 if i == 0 else 1) / fact#(math.factorial(n-1)*2**(n-1))

    def contract_indices(self, i : int, j: int, major = None, minor = None): #major defuaults to Nx + 1 minor to Ny + 1
        if major == None:
            major = self.Nx + 1
        if minor == None:
            minor = self.Ny + 1
        assert(i <= major and j <= minor)

        return i + major * j

    def expand_indices(self, i: int, major = None, minor = None): 
        if major == None:
            major = self.Nx + 1
        if minor == None:
            minor = self.Ny + 1
        assert(i < major * minor)

        return i % major, int(i / major)

    def __call__(self, x, y):
        sum = 0.0
        Y = y/self.P0
        X = (x/self.a0 - self.Q3)/self.Q0
        for j in range(self.Ny2):
            for i in range(self.Nx + 1):
                sum += self.vec[self.contract_indices(i, j)] * chebyt(i)(X) * chebyt(2 * j)(Y)
        return sum

