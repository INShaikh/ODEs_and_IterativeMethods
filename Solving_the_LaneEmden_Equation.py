import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#####################
# Functions
#####################

def matrix_multiplyer(a, b, c, u):
    
    # Define 2d array
    d_array = np.zeros((len(a), len(a) + 2))
    
    # Populate array
    for i in range (0, len(a)):
        d_array[i, i] = a[i]
        d_array[i, i + 1] = b[i]
        d_array[i, i + 2] = c[i]
        
    # Trim  and multiply arrays
    trimmed_array = d_array[ : , 1 : -1]
    multiplied_matrix = np.dot(trimmed_array, u)    
    
    return multiplied_matrix


def make_arrays_LHS(xi, xf, Nx, solution_guess):
    dx = (xf-xi) / Nx
    x = np.linspace(xi - (dx / 2), xf + (dx / 2), Nx + 2)
    
    # First matrix values
    a1 = np.ones(Nx - 1) / dx**2
    b1 = np.ones(Nx - 1) * -2. / dx**2
    c1 = np.ones(Nx - 1) / dx**2
    
    # First matrix boundary conditions
    b1[Nx - 2] = (-3.) / (dx**2)
    
    # Matrix multiplication
    first_term = matrix_multiplyer(a1, b1, c1, solution_guess)
    
    # Add seond part of matrix
    first_term[0] += (1 / dx**2)
    
    # Second matrix values
    a2 = np.ones(Nx - 1) * -1 / (dx * x[2:-1])
    b2 = np.zeros(Nx - 1)
    c2 = np.ones(Nx - 1) / (dx * x[2:-1]) 
    
    # Second matrix boundary conditions
    b2[Nx - 2] = -1 / (dx * x[-2])
    
    # Matrix multiplication
    secon_term = matrix_multiplyer(a2, b2, c2, solution_guess)
    
    # Add seond part of matrix
    secon_term[0] += (-1 / (dx * x[2]))
    
    # Third matrix values
    third_term  = solution_guess**2
    
    # Summate all the matrices
    F = first_term + secon_term + third_term
    
    return  F, x, dx


def A(dx, x, solution_guess):
    
    # First matrix values
    a1 = np.ones(Nx - 1) / dx**2
    b1 = np.ones(Nx - 1) * -2. / dx**2
    c1 = np.ones(Nx - 1) / dx**2
    
    # First matrix boundary conditions
    b1[Nx - 2] = (-3.) / (dx**2)

    # Second matrix values
    a2 = np.ones(Nx - 1) * -1 / (dx * x[2:-1])
    b2 = np.zeros(Nx - 1)
    c2 = np.ones(Nx - 1) / (dx * x[2:-1]) 
    
    # Second matrix boundary conditions
    b2[Nx - 2] = -1 / (dx * x[-2])

    # Third matrix values
    a3 = np.zeros(Nx - 1)
    b3 = solution_guess * 2
    c3 = np.zeros(Nx - 1)

    a_total = a1 + a2 + a3
    b_total = b1 + b2 + b3
    c_total = c1 + c2 + c3
    
    return a_total, b_total, c_total


def tri_diag_method(a, b, c, f):
    
    # Create needed arrays
    soln = np.zeros(len(a))
    dprime = np.zeros(len(a))
    cprime = np.zeros(len(a))
    
    # Set initial conditions
    cprime[0] = c[0] / b[0]
    dprime[0] = f[0] / b[0]
    
    # Define cprime and dprime arrays
    for i in range (1, len(a)):
        cprime[i] = c[i] / (b[i] - (cprime[i-1] * a[i]))
        dprime[i] = (f[i] - (dprime[i-1] * a[i])) / (b[i] - (cprime[i-1] * a[i]))
    
    # Solve function using back solver
    soln[len(a) - 1] = dprime[len(a) - 1]
    
    for i in range(len(a) - 2, -1, -1):
        soln[i] = dprime[i] - cprime[i] * soln[i + 1]
        
    return soln


def Newton_Raphson(xi, xf, Nx, solution_guess, repeat_max, tolerance):


    # Create array to write solutions into
    solutions = np.zeros((repeat_max, Nx + 2))
    delta = np.zeros(repeat_max)
    
    # Create loop to carry out newton raphson iteration
    for i in range(0, repeat_max):
        
        # Evaluate array F and return the staggered x array
        F, x, dx = make_arrays_LHS(xi, xf, Nx, solution_guess[2:-1])
        
        # Find the A array
        a, b, c = A(dx, x, solution_guess[2:-1])
        
        # Use tri-diagonal method to find the d value
        d = tri_diag_method(a, b, c, F)
        
        # Update solution arrays
        solution_guess[2:-1] = solution_guess[2:-1] - d
        solutions[i,:] = solution_guess[:]
        delta[i] = np.sqrt(sum(j**2 for j in d))
        if delta[i] < tolerance:
            break 

    return x, dx, solutions[i,:], delta 


def Convergance_Rate(In, I2n, I4n):
    return np.abs(In - I2n), np.abs(I2n - I4n)

#####################
# Main programme
#####################

# Initial parameters
xi=0.
xf=4.35			
Nx=100
yi = 1.
yf = 0.
yi_prime = 0
repeat_max = 30
tolerance = 10**(-5)

# Initial guess array
solution_guess = np.ones(Nx + 2)
solution_guess[Nx + 1] = yf

x_100, dx_100, solution_100, delta_100  = Newton_Raphson(xi, xf, Nx, solution_guess, repeat_max, tolerance)


# Plot out function
plt.figure()
plt.title('Solution for the lane Emdem Equation')
plt.plot(x_100, solution_100, label='Polytropic Index $n=2$')
plt.xlabel('$Xi$')
plt.ylabel('$Theta$')
plt.legend(loc='best')
plt.grid()
plt.show()


########################
# Convergence and Order
########################
# Initial extra parameters
Nx = 200

# Initial guess array
solution_guess = np.ones(Nx + 2)
solution_guess[Nx + 1] = yf

x_200, dx_200, solution_200, delta_200  = Newton_Raphson(xi, xf, Nx, solution_guess, repeat_max, tolerance)

# Initial extra parameters
Nx = 400

# Initial guess array
solution_guess = np.ones(Nx + 2)
solution_guess[Nx + 1] = yf

x_400, dx_400, solution_400, delta_400  = Newton_Raphson(xi, xf, Nx, solution_guess, repeat_max, tolerance)


# Find the interpolated version of the solutions
solution_200_inter_func = interpolate.interp1d(x_200, solution_200, 'cubic')
solution_200_inter = solution_200_inter_func(x_100[2:-1])

solution_400_inter_func = interpolate.interp1d(x_400, solution_400, 'cubic')
solution_400_inter = solution_400_inter_func(x_100[2:-1])

# Plot out the interpolated functions to show similarity
plt.figure()
plt.title('Solution for the lane Emdem Equation for Different N using Interpolation')
plt.plot(x_100[2:-1], solution_100[2:-1], label='$N = 100$')
plt.plot(x_100[2:-1], solution_200_inter, label='$N = 200$')
plt.plot(x_100[2:-1], solution_400_inter, label='$N = 400$')
plt.xlabel('$Xi$')
plt.ylabel('$Theta$')
plt.legend(loc='best')
plt.grid()
plt.show()

# Calculate convergance order
convergance_rate1, convergance_rate2 = Convergance_Rate(solution_100[2:-1], solution_200_inter, solution_400_inter)

# Plot out convergance
plt.figure()
plt.title('Convergance Rate')
plt.plot(x_100[2:-1], convergance_rate1, label='top')
plt.plot(x_100[2:-1], convergance_rate2, label='bottom')
plt.xlabel('$Xi$')
plt.ylabel('Convergance Rate')
plt.legend(loc='best')
plt.grid()
plt.show()

############################
# New Solution
############################

# Initial extra parameters
tolerance = 10**(-20)

# Initial guess array
solution_guess = np.ones(Nx + 2)
solution_guess[Nx + 1] = yf

x_400_lt, dx_400_lt, solution_400_lt, delta_400_lt  = Newton_Raphson(xi, xf, Nx, solution_guess, repeat_max, tolerance)

# Plot out new solution with decreased tolerance
plt.figure()
plt.title('Solution for the lane Emdem Equation $(N=400)$')
plt.plot(x_400_lt[2:-1], solution_400_lt[2:-1], label='Tolerance = 10^$(-20)$')
plt.xlabel('$Xi$')
plt.ylabel('Theta')
plt.legend(loc='best')
plt.grid()
plt.show()

delta_x = np.linspace(1,np.size(delta_400_lt),np.size(delta_400_lt))

# Plot out the delta of the new solution with decreased tolerance
plt.figure()
plt.title('Delta of iteration of Solution for the lane Emdem Equation $(N=400)$')
plt.semilogy(delta_x, delta_400_lt, label='Tolerance = 10^$(-20)$')
plt.xlabel('Iteration Number')
plt.ylabel('Delta')
plt.legend(loc='best')
plt.grid()
plt.show()