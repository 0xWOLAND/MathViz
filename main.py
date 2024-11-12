import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import griddata

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

N = 20  
D = 10
INTERP_N = 100  # Interpolated points
INTERP_D = 50

# Create ranges for the mesh
number_range = (1, 10000)
d_range = (2, 20)

# Create original points
numbers = np.linspace(*number_range, N, dtype=int)
d_values = np.linspace(*d_range, D, dtype=int)

# Create finer grid for interpolation
numbers_fine = np.linspace(*number_range, INTERP_N)
d_values_fine = np.linspace(*d_range, INTERP_D)

# Pre-calculate all final values
final_steps = np.zeros((N, D))
for i in range(N):
    for j in range(D):
        _, steps = appx_squaring(numbers[i], d_values[j])
        final_steps[i, j] = max(0, steps)

# Create points for interpolation
points = np.array([(numbers[i], d_values[j]) 
                   for i in range(N) 
                   for j in range(D)])
values = final_steps.flatten()

# Create fine meshgrid for interpolation using actual values
X_fine, Y_fine = np.meshgrid(numbers_fine, d_values_fine)

# Interpolate final values
final_steps_fine = griddata(points, values, (X_fine, Y_fine), method='cubic')

# Initialize current state
current_steps_fine = np.zeros_like(final_steps_fine)

# Create figure
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    
    # Update current_steps
    mask = current_steps_fine < final_steps_fine
    current_steps_fine[mask] += 0.2
    
    # Plot smooth surface
    surf = ax.plot_surface(X_fine, Y_fine, current_steps_fine,
                          cmap='viridis',
                          linewidth=0,
                          antialiased=True,
                          alpha=0.8)
    
    # Find current maximum position
    max_pos = np.unravel_index(np.argmax(current_steps_fine), current_steps_fine.shape)
    max_value = current_steps_fine[max_pos]
    n_max = X_fine[max_pos]
    d_max = Y_fine[max_pos]
    
    # Plot maximum point with vertical line
    ax.scatter([n_max], [d_max], [max_value], 
               color='red', s=100, marker='*')
    ax.plot([n_max, n_max], [d_max, d_max], [0, max_value], 
            color='red', linestyle='--', alpha=0.5)
    
    # Add text annotation for maximum point (offset for visibility)
    ax.text(n_max, d_max, max_value, 
            f'  Max: {max_value:.1f} steps\n  N: {int(n_max)}\n  d: {int(d_max)}',
            color='white', fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Numerator')
    ax.set_ylabel('Denominator')
    ax.set_zlabel('Steps')
    ax.set_title('Steps Required for Different Numbers and d Values')
    
    # Set axis limits
    ax.set_xlim(number_range)
    ax.set_ylim(d_range)
    ax.set_zlim(0, np.max(final_steps_fine) * 1.1)
    
    # Set custom tick labels
    ax.set_xticks(np.linspace(*number_range, 5))
    ax.set_yticks(np.linspace(*d_range, 5))
    
    # Set view angle
    ax.view_init(elev=20, azim=frame/2)
    
    # Print progress
    completed = np.sum(current_steps_fine >= final_steps_fine)
    total = INTERP_N * INTERP_D
    print(f"\rProgress: {completed}/{total} complete ({completed/total*100:.1f}%) | "
          f"Current max: {max_value:.1f} steps at N={int(n_max)}, d={int(d_max)}", end="")
    
    return [surf]

# Create animation
anim = animation.FuncAnimation(
    fig,
    update,
    frames=None,
    interval=50,
    blit=False
)

plt.tight_layout()
plt.show()