import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Nx, Ny = 60, 15  # Number of grid points
Lx, Ly = 1.0, 1.0 / 4  # Domain size
hx, hy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
mu = 1.0  # Viscosity

# Initialize fields
p = np.zeros((Nx, Ny))  # Pressure
u = np.zeros((Nx, Ny))  # Velocity u
v = np.zeros((Nx, Ny))  # Velocity v (zero everywhere by assumption)

# Time-stepping parameters
dt = 0.01  # Time step
num_steps = 200  # Number of time steps

# prepare graphic interface:
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(-Ly/2, Ly/2, Ny))
Lmax=np.max([Lx,Ly])
fig,ax=plt.subplots(figsize=(12*Lx/Lmax, 12*Ly/Lmax))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Pressure and Velocity Field')
# Backward Euler iteration loop
for step in range(num_steps):
    p_new = p.copy()
    # refresh graphic:
if step % (num_steps // 10) == 0:
        #fig, ax = plt.subplots(figsize=(8, 4))
        ax.contourf(X, Y, p.T, levels=50, cmap='coolwarm')
        fig.colorbar(ax.contourf(X, Y, p.T, levels=50, cmap='coolwarm'), ax=ax, label='Pressure')
        ax.quiver(X, Y, u.T, v.T, scale=10, angles='xy')
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_title(f'Iteration {step}')
        plt.pause(0.1)
        fig.clf()
    # Solve pressure Poisson equation (iterative Gauss-Seidel update)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            p_new[i, j] = (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1]) / 4
    
    # Apply boundary conditions
    p_new[0, :] = 1  # Left
    p_new[-1, :] = 0  # Right
    
    # Solve Stokes equation for velocity u
    u_new = u.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u_new[i, j] = (dt / mu) * (
                -(p_new[i+1, j] - p_new[i-1, j]) / (2 * hx) +
                (v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1]) / 4
            )
    
    # Apply no-slip conditions on velocity
    u_new[:, 0] = 0
    u_new[:, -1] = 0
    
    # Update fields
    p = p_new.copy()
    u = u_new.copy()
