import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the Rosenbrock function
a = 1
b = 100

# Rosenbrock function
def rosenbrock(x, y):
    return (a - x)**2 + b * (y - x**2)**2

# Create grid for x and y
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Set up 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with lower opacity
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.15)  # Lower alpha for surface

# Highlight points with larger size, increased alpha, and contrasting colors
# Diamond point for iteration #1
points_x = [-0.35086]
points_y = [0.09]
points_z = [rosenbrock(points_x[0], points_y[0])]
ax.scatter(points_x, points_y, points_z, color='yellow', s=50, edgecolor='black', depthshade=False, label='Melhor solução iteração #1', marker='D', alpha=0.9)

# Circle point for iteration #15
points_x = [0.24605]
points_y = [0.05]
points_z = [rosenbrock(points_x[0], points_y[0])]
ax.scatter(points_x, points_y, points_z, color='green', s=50, edgecolor='black', depthshade=False, label='Melhor solução iteração #15', marker='o', alpha=0.9)

# Triangle point for iteration #25
points_x = [1.0]
points_y = [0.96]
points_z = [rosenbrock(points_x[0], points_y[0])]
ax.scatter(points_x, points_y, points_z, color='black', s=50, edgecolor='black', depthshade=False, label='Melhor solução iteração #25', marker='^', alpha=0.9)

# Square point for iteration #29
points_x = [0.99815]
points_y = [0.9568]
points_z = [rosenbrock(points_x[0], points_y[0])]
ax.scatter(points_x, points_y, points_z, color='red', s=50, edgecolor='black', depthshade=False, label='Melhor solução iteração #29', marker='s', alpha=0.9)

# Star point for the global minimum
ax.scatter([1.], [1.], [rosenbrock(1.0, 1.0)], color='blue', s=150, edgecolor='black', depthshade=False, label='Mínimo Global', marker='*', alpha=0.9)

# Labels and title
ax.set_title('Função de Rosenbrock')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()

plt.show()

