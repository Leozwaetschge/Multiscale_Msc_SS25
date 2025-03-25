from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Define the lattice size
Nx = 5  # Number of columns in the lattice
Ny = 5  # Number of rows in the lattice
N = Nx * Ny  # Total number of spins in the lattice

# Define physical constants
J = 1  # Ferromagnetic exchange constant
D = 2.1*J  # Dzyaloshinskii-Moriya (DM) interaction constant
B = 0.25*D**2/J  # Magnetic field strength

def index_to_position(index):
    """
    Convert a spin index to its 2D position (row, column) in the lattice.
    """
    row = index // Nx
    col = index % Nx
    return np.array([row, col])

def position_to_index(position):
    """
    Convert a 2D position (x, y) to a spin index.
    """
    return int(position[1] * Nx + position[0])

# Create a list of neighbors for each spin, considering periodic boundary conditions
neighbor_indices = [[] for _ in range(N)]
neighbor_vectors = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # Relative positions of neighbors

for spin_index in range(N):
    spin_position = index_to_position(spin_index)
    for neighbor_vector in neighbor_vectors:
        neighbor_position = spin_position + neighbor_vector
        # Apply periodic boundary conditions
        neighbor_position[0] = neighbor_position[0]  % Ny  # Periodic in rows
        neighbor_position[1] = neighbor_position[1]  % Nx  # Periodic in columns
        neighbor_indices[spin_index].append(position_to_index(neighbor_position))

def angles_to_vector(angles):
    """
    Convert spherical coordinates (theta, phi) to a normalized 3D vector.
    """
    x = np.sin(angles[0]) * np.cos(angles[1])
    y = np.sin(angles[0]) * np.sin(angles[1])
    z = np.cos(angles[0])
    return np.array([x, y, z])

def angles_list_to_vectors(angles_list):
    """
    Convert a list of spherical coordinates (theta_1, phi_1, theta_2, ...) to a list of 3D vectors.
    """
    vectors = np.zeros((N, 3))
    for index in range(N):
        theta = angles_list[2 * index]
        phi = angles_list[2 * index  + 1]
        vectors[index] = angles_to_vector([theta, phi])
    return vectors

def compute_energy(angles_list):
    """
    Compute the energy of a spin configuration given by a list of spherical coordinates (theta_1, phi_1, theta_2, ...).
    """
    energy = 0

    # Single-ion terms (magnetic field and anisotropy)
    for index in range(N):
        theta = angles_list[2 * index]
        energy += -B * np.cos(theta)  # Magnetic field contribution

    # Interaction terms (exchange and DM interactions)
    for index1 in range(N):
        spin1 = angles_to_vector(angles_list[2 * index1:2 * index1 + 2])
        for neighbor_index in range(len(neighbor_indices[index1])):
            index2=neighbor_indices[index1][neighbor_index]
            spin2 = angles_to_vector(angles_list[2 * index2:2 * index2 + 2])
            energy -= 0.5* J * np.dot(spin1,spin2) #to complete with half ot the exchange energy between spin1 and spin2
            
            energy += 0.5* D * np.dot(XXXXXXX,np.cross(spin2,spin1)) #to complete with half of the DM energy between spin1 and spin2
    return energy

def plot_spin_configuration(angles_list, scale=1):
    """
    Plot the spin configuration using quiver plot.
    """
    # Convert spherical coordinates to 3D vectors
    spin_vectors = angles_list_to_vectors(angles_list)

    # Create the lattice points
    x_positions, y_positions = np.meshgrid(np.arange(Nx), np.arange(Ny))
    x_positions = x_positions.flatten() - spin_vectors[:, 0] / (scale * 2)  # Center the arrows
    y_positions = y_positions.flatten() - spin_vectors[:, 1] / (scale * 2)  # Center the arrows

    # Plot the spins
    plt.figure()
    plt.quiver(x_positions, y_positions, spin_vectors[:, 0], spin_vectors[:, 1], spin_vectors[:, 2], angles='xy', scale_units='xy', width=0.03, scale=scale, cmap='viridis')
    plt.xlim(-1, Nx)
    plt.ylim(-1, Ny)
    cbar = plt.colorbar()
    cbar.set_label(r'$S^z$')
    plt.show()

# Initialize the spin configuration with a single spin flipped
initial_angles = np.zeros(2 * N)
initial_angles[2 * (N // 2)] = np.pi  # Flip a middle spin  

# Plot the initial configuration
plot_spin_configuration(initial_angles)

# Minimize the energy to find a stable state
result = minimize(compute_energy, initial_angles)

# Plot the minimized configuration
plot_spin_configuration(result.x)
print("final energy: %1.3f"%result.fun)

