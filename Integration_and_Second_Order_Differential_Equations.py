import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter


###############################################################################
# PART A
###############################################################################

# Read in picture.
im = imageio.imread('Picture1.JPG')

# Process image.
image=im[:,:,0]

# Plotting and saving the processed image.
fig1, ax1 = plt.subplots()
ax1.imshow(image,cmap='gray')
imageio.imwrite('Picture1_greyscale.jpeg', image)

# Find the size of the array of the filtered Picture1.
dimx=image.shape[0]
dimy=image.shape[1]

# Produce a 2d array of zeroes the same size as that of Picture1.
Result=np.zeros((dimx,dimy))

# Apply gaussian filter on the filtered Picture1.
imsm=gaussian_filter(image,sigma=5.)

# Plotting and saving the smoothed image.
fig2, ax2 = plt.subplots()
ax2.imshow(imsm,cmap='gray')
imageio.imwrite('Picture1_gaussfilt.jpeg', imsm) 

'''
dx=1
dy=1

for i in range(1,dimx-2):
    for j in range(1,dimy-2):
        num = ((imsm[i+1,j] - imsm[i-1,j])/(2))**2
        den = ((imsm[i,j+1] - imsm[i,j-1])/(2))**2
        Result[i,j] = np.sqrt(num/den)
'''


###############################################################################
# PART B
###############################################################################

# Define functions needed.
def simpsrule(f, a, b, N):
    
    # Create an xValue array.
    xArray = np.linspace(a, b, N + 1)
    
    # Define variables.
    deltax = (b - a) / N
    summation = 0
    
    # For-loop to calculate sum then return value.
    for i in range (1, np.int(N/2) + 1):
        summation += f(xArray[2*i - 2]) + 4*f(xArray[2*i - 1]) + f(xArray[2*i])  
    return (deltax / 3) * summation


# Function to integrate.
def f1(x):
	return 5 * (x**2) + 2 * (x**5)

# Define variables.
a=0.
b=2.
N = np.zeros(10)
y = np.zeros(10)
R = np.zeros(10)

# For-loop to calculate convergance rate as a function of N intervals.
for i in range (1, 11):
    N[i - 1] = 2**i
    y[i - 1] = simpsrule(f1, a, b, 2**i)
    
    var1 = (simpsrule(f1, a, b, (2**i)) - simpsrule(f1, a, b, (2**i) * 2))
    var2 = (simpsrule(f1, a, b, (2**i) * 2) - simpsrule(f1, a, b, (2**i) * 4))
    R[i - 1] = var1 / var2

# Plot results.
plt.figure()
plt.plot(N, R, label = '$f(x) = 5x^2 + 2x^5$')
plt.xlabel('$N$ $Intervals$ '); plt.ylabel('Convergence Rate')
plt.legend(loc='best')
plt.grid();
plt.title("Plot of Convergance Rate Against N Intervals for Simpson's Rule")
plt.savefig('Plot_Of_Convergance_Rate_Against_Intervals_(Simpsons).png')

##############################################################################

# New function to integrate.
def f2(x):
	return 3 * (x**2) + 4 * (x**3)

# Define new variables.
a = 0.
b = 2.
N1 = np.zeros(10)
y1 = np.zeros(10)
R1 = np.zeros(10)
y_error = np.zeros(10)
y_analytic = b**3 + b**4

# For-loop to calculate error as a function of N intervals.
for i in range (1, 11):
    N1[i - 1] = 2**i
    y1[i - 1] = simpsrule(f2, a, b, 2**i)
    y_error[i - 1] = np.abs(y1[i - 1] - y_analytic)

# PLot results.
plt.figure()
plt.plot(N, y_error, label = '$f(x) = 3x^2 + 4x^3$')
plt.xlabel('$N$ $Intervals$ '); plt.ylabel('Error')
plt.legend(loc='best')
plt.grid();
plt.title("Plot of Error Against N Intervals for Simpson's Rule")
plt.savefig('Plot_Of_Error_Against_Intervals_(Simpsons).png')


###############################################################################
# PART C
###############################################################################

# Function for rhs of ODE harmonic oscillator y'' + (w**2)*y = 0.
def fpp(y):
    return -(w**2)*y

# Euler integration step function.
def Euler(yn, rhs, dt):
    return yn + rhs * dt

# ODE function solver to return time and y values.
def ODEsolve(Tmax, N, rhs, method, ic):
    
    # Calculating variables.
    t=np.zeros(N+1)
    yn=np.zeros(N+1)
    ynp=np.zeros(N+1)
    t[0] = ic[0]
    yn[0] = ic[1]
    ynp[0] = ic[2]
    
    # Calculating dt.
    dt = (Tmax - t[0]) / N    
    
    # For-loop to find Euler integrals.
    for i in range (0 , N ):   
        t[i + 1] = t[i] + dt
        ynp[i + 1] = method(ynp[i], rhs(yn[i]), dt)        
        yn[i + 1] = method(yn[i], ynp[i], dt)
        
    return t, yn


# Define variables.
w=2*np.pi
N=100

# Calculating y values for the harmonic oscillator when N = 100.
t1, yn1 = ODEsolve(1 , N, fpp, Euler, np.array([0,1,0]))

# Plot results.
plt.figure()
plt.plot(t1,yn1,'b-', label='harmonic oscillator')
plt.xlabel (r'$time$ $s$ '); plt. ylabel (r'$y$ $position$')
plt.grid ();
plt.title("$y''+w^2y=0$ $Euler$ $Method$ $N=100$")

###############################################################################

# Creating variables.
E = np.zeros(10)
Narray = np.zeros(10)
mew_f = 10**(-8)
T = 1

# For-loop to create data for harmonic oscillator with time intervals 2**i
# such that N = 2, 4, 8,...,1024.
for i in range (1,11):
    Narray[i-1] = 2**i
    t1, yn1 = ODEsolve(1, 2**i, fpp, Euler, np.array([0,1,0]))
    
    # Calculating errors.
    dt = t1[1] - t1[0]
    E[i - 1] = ((Narray[i - 1] * mew_f) + (T * dt))


# Create log-log plot of the error of a harmonic oscillator using Euler's
# method as a function of N.
plt.figure()
plt.loglog( Narray, E ,'b-')    
plt.xlabel (r'$N$ Intervals'); plt. ylabel (r'$Error$')
plt.grid ()
plt.title("Log-Log Plot of the Error of a Harmonic Oscillator using Eulers Method as a Function of N")