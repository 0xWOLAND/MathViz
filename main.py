import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def appx_squaring(n, d):
    steps = 0
    try:
        while n % d:
            n *= (n + d + 1) // d
            steps += 1
            if steps > 1000:  # Safety limit
                return (-1, -1)
        return (n, steps)
    except (OverflowError, ValueError):
        return (-1, -1)

# Initialize data (smaller for testing)
N = 20  # Number of x values
D = 10  # Number of d values

# Create evenly spaced values
numbers = np.linspace(1, 1000, N, dtype=int)
d_values = np.linspace(2, 20, D, dtype=int)

# Pre-calculate all final values
final_steps = np.zeros((N, D))
for i in range(N):
    for j in range(D):
        _, steps = appx_squaring(numbers[i], d_values[j])
        final_steps[i, j] = max(0, steps)  # Convert -1 to 0 for visualization

# Initialize current state
current_steps = np.zeros((N, D))

# Create figure
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid
X, Y = np.meshgrid(np.arange(N), np.arange(D))

def update(frame):
    ax.clear()
    
    # Update current_steps
    mask = current_steps < final_steps
    current_steps[mask] += 0.2
    
    # Plot surface
    surf = ax.plot_surface(X, Y, current_steps.T, 
                          cmap='viridis',
                          linewidth=0,
                          antialiased=True)
    
    # Set labels
    ax.set_xlabel('Number Index')
    ax.set_ylabel('d Value')
    ax.set_zlabel('Steps')
    ax.set_title('Steps Required for Different Numbers and d Values')
    
    # Set view angle
    ax.view_init(elev=20, azim=frame/2)
    
    # Print progress
    completed = np.sum(current_steps >= final_steps)
    total = N * D
    print(f"\rProgress: {completed}/{total} complete ({completed/total*100:.1f}%)", end="")
    
    return [surf]

# Create animation
anim = animation.FuncAnimation(
    fig,
    update,
    frames=None,
    interval=50,
    blit=False
)

plt.show()