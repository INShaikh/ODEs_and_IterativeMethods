from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


#############################################################################
# Part A
#############################################################################

# Define the initial conditions of the wave equation PDE.
def initialConditions(xMin, xMax, Nx, dx):
    
    xArray = np.linspace(xMin-dx, xMax, Nx + 2)
    
    uInitial = np.exp((-xArray**2) / (2.*sigma**2))
    vInitial = 0
    wInitial = -(xArray/(sigma**2))*uInitial

    return uInitial, vInitial, wInitial
    
    
# Define the boundary conditions of the wave equation PDE.
def boundaryConditions(uArray, vArray, wArray, Nx, timeIndex):
    
    uArray[0, timeIndex] = uArray[Nx, timeIndex]
    uArray[Nx + 1, timeIndex] = uArray[1, timeIndex]
    
    vArray[0, timeIndex] = vArray[Nx, timeIndex]
    vArray[Nx + 1, timeIndex] = vArray[1, timeIndex]
    
    wArray[0, timeIndex] = wArray[Nx, timeIndex]
    wArray[Nx + 1, timeIndex] = wArray[1, timeIndex]
    
    return uArray, vArray, wArray
    

# Define the RHS function of the wave equation PDE.
def RHS(uArray, vArray, wArray, Nx, dx):
    
    # Define arrays which are derivatives with respect to time.
    uTderivative = np.zeros(Nx + 2)
    vTderivative = np.zeros(Nx + 2)
    wTderivative = np.zeros(Nx + 2)
    
    # Compute the derivative values using the right hand sides of the 
    # first order coupled equations: u/dt = v, v/dt = w/dx and w/dt = v/dx.
    uTderivative = vArray
    vTderivative[1 : Nx + 1] = (wArray[2 : Nx + 2] - wArray[0 : Nx]) / (2*dx)
    wTderivative[1 : Nx + 1] = (vArray[2 : Nx + 2] - vArray[0 : Nx]) / (2*dx)

    return uTderivative, vTderivative, wTderivative
    

# Define the Euler function solver.
def Euler(uArray, vArray, wArray, Nx, dx, dt, function):
    
    # Produce the RHS arrays.
    RHSuArray, RHSvArray, RHSwArray = RHS(uArray, vArray, wArray, Nx, dx)
    
    # Apply Euler's method to each of the first order coupled variables.
    uArray = uArray + RHSuArray * dt
    vArray = vArray + RHSvArray * dt
    wArray = wArray + RHSwArray * dt
    
    return uArray, vArray, wArray
    

# Define the main function to solve the wave equation.
def MainWaveSolver(xMin, xMax, Ti, Tf, Nx, method):
    
    # Calculating variables.
    dx = (xMax - xMin) / Nx
    dt = courantFactor * dx
    Nt = np.int((Tf - Ti) / dt)
    
    # Creating the position and time arrays.
    xArray = np.linspace(xMin - dx, xMax, Nx + 2)
    tArray = np.linspace(Ti, Tf, Nt + 1)

    # Create empty arrays to fill for the first-order couple equations.
    u = np.zeros((Nx + 2, Nt + 1))
    v = np.zeros((Nx + 2, Nt + 1))
    w = np.zeros((Nx + 2, Nt + 1))

    # Setting the initial conditions.
    u[0:Nx + 2, 0], v[0:Nx + 2, 0], w[0:Nx + 2, 0] = initialConditions(xMin, xMax, Nx, dx)
    timeStep = 0
    
    u, v, w = boundaryConditions(u, v, w, Nx, timeStep)
    
    # Performing the integration steps.
    for i in range(1, Nt + 1):
        
        # Euler method selection.
        #if method=='Euler':		
        u[:,i], v[:,i], w[:,i] = method(u[:, i - 1], v[:, i - 1], w[:, i - 1], Nx, dx, dt, RHS)
			
        u, v, w = boundaryConditions(u, v, w, Nx, i)
				
    return tArray, xArray, u


# Define the given variables of part A.
xMin = -1
xMax = 1
Nx = 100

Ti = 0
Tf = 1.2
courantFactor = 0.5
sigma = 0.1


# Perform function call to Euler solver.
timeArray, positionArray, solutionArray = MainWaveSolver(xMin,
                                                         xMax,
                                                         Ti,
                                                         Tf,
                                                         Nx,
                                                         Euler)

# 2D plot of x and t with varying intensity.
plt.figure()
plt.imshow(solutionArray)

# Plot solution for times t = 0.0, t = 0.5 and t = 1.0
plt.figure()
plt.xlim(-1, 1)
plt.plot(positionArray, solutionArray[:, 0], label = 't = 0.0s')
plt.plot(positionArray, solutionArray[:, 50], label = 't = 0.5s')
plt.plot(positionArray, solutionArray[:, 100], label = 't = 1.0s')

plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('Plot of Wave Equation for N = 100 at Various Times Using Euler Method')
plt.legend(loc = 'best')


#############################################################################
# Part B
#############################################################################

# Define the Runge-Kutta 4th order method for each of the first order coupled
# equations for the wave function PDE.
def RungeKutta4thOrder(uArray, vArray, wArray, Nx, dx, dt, function):
    
    rhs_uArray1, rhs_vArray1, rhs_wArray1 = RHS(uArray, vArray, wArray, Nx, dx)     
    uArray_k1 = rhs_uArray1
    vArray_k1 = rhs_vArray1
    wArray_k1 = rhs_wArray1
    
    rhs_uArray2, rhs_vArray2, rhs_wArray2 = RHS(uArray_k1, vArray_k1, wArray_k1, Nx, dx)  
    uArray_k2 = uArray + rhs_uArray2*(dt/2)
    vArray_k2 = vArray + rhs_vArray2*(dt/2)
    wArray_k2 = wArray + rhs_wArray2*(dt/2)
    
    rhs_uArray3, rhs_vArray3, rhs_wArray3 = RHS(uArray_k2, vArray_k2, wArray_k2, Nx, dx)
    uArray_k3 = uArray + rhs_uArray3*(dt/2)
    vArray_k3 = vArray + rhs_vArray3*(dt/2)
    wArray_k3 = wArray + rhs_wArray3*(dt/2)
    
    rhs_u3, rhs_v3, rhs_w3 = RHS(uArray_k3, vArray_k3, wArray_k3, Nx, dx)
    uArray_k4 = uArray + rhs_uArray3*dt
    vArray_k4 = vArray + rhs_vArray3*dt
    wArray_k4 = wArray + rhs_wArray3*dt
    
    uArray_RK = uArray + (dt/6) * (uArray_k1 + 2*uArray_k2 + 2*uArray_k3 + uArray_k4)
    vArray_RK = vArray + (dt/6) * (vArray_k1 + 2*vArray_k2 + 2*vArray_k3 + vArray_k4)
    wArray_RK = wArray + (dt/6) * (wArray_k1 + 2*wArray_k2 + 2*wArray_k3 + wArray_k4)	
        
    return uArray_RK, vArray_RK, wArray_RK

# Define Variables and call main function solver
Nx = 100

timeArrayRK100, positionArrayRK100, solutionArrayRK100 = MainWaveSolver(xMin,
                                                                        xMax,
                                                                        Ti,
                                                                        Tf,
                                                                        Nx,
                                                                        RungeKutta4thOrder)

# 2D plot of x and t with varying intensity.
plt.figure()
plt.imshow(solutionArray)

# Plot solution for times t = 0.0, t = 0.5 and t = 1.0
plt.figure()
plt.xlim(-1, 1)
plt.plot(positionArrayRK100, solutionArrayRK100[:, 0], label = 't = 0.0s')
plt.plot(positionArrayRK100, solutionArrayRK100[:, 50], label = 't = 0.5s')
plt.plot(positionArrayRK100, solutionArrayRK100[:, 100], label = 't = 1.0s')

plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('Plot of Wave Equation for N = 100 at Various Times Using Runge-Kutta Method')
plt.legend(loc = 'best')


#############################################################################
# Part C
#############################################################################

# Producing RK4 solutions for N = 200 and N = 400.
Nx = 200
timeArrayRK200, positionArrayRK200, solutionArrayRK200 = MainWaveSolver(xMin,
                                                                        xMax,
                                                                        Ti,
                                                                        Tf,
                                                                        Nx,
                                                                        RungeKutta4thOrder)

Nx = 400
timeArrayRK400, positionArrayRK400, solutionArrayRK400 = MainWaveSolver(xMin,
                                                                        xMax,
                                                                        Ti,
                                                                        Tf,
                                                                        Nx,
                                                                        RungeKutta4thOrder)

# Define a function to create L2norm arrays
def L2norm(u1,u2,Nx1,Nt, arrayJumpFactor):
	suma=np.zeros(Nt + 1)
	for i in range(Nt + 1):
		suma[i]=np.sum((u1[1:,i]-u2[1::arrayJumpFactor, 2 * i])**2)
	
	return np.sqrt(suma/Nx1)

L2arrayC1 = L2norm(solutionArrayRK200, solutionArrayRK400, 200, timeArrayRK200.shape[0] - 1, 2)
L2arrayC2 = L2norm(solutionArrayRK100, solutionArrayRK200, 100, timeArrayRK100.shape[0] - 1, 2)


# Plot solution for times t = 0.0, t = 0.5 and t = 1.0
plt.figure()
plt.plot(timeArrayRK200, L2arrayC1, label = 'N = 400')
plt.plot(timeArrayRK100, L2arrayC2, label = 'N = 200')

# Graph formatting
plt.xlabel('Time (s)')
plt.ylabel('L2 Norm')
plt.title('L2 Norm for the Runge-Kutta 4th Order Method of various N values')
plt.legend(loc = 'best')


# This shows the RK4 method in this case to be of order 1, this is explained
# by the fact that the RK4 method was used to integrate first order equations,
# rather than a fourth order or higher function.
plt.figure()
plt.plot(timeArrayRK200, L2arrayC1 * 2, label = 'N = 400')
plt.plot(timeArrayRK100, L2arrayC2, label = 'N = 200')
plt.xlabel('Time')
plt.ylabel('L2 Norm')
plt.xscale('log')
plt.ylim(10e-6, 10e1)
plt.yscale('log')
plt.title('Log-Log Plot of Runge-Kutta 4th Order Method with Factor Comparison')
plt.legend(loc = 'best')
