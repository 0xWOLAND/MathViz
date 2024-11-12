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

# Initialize data
N = 20  # Original points
D = 10
INTERP_N = 100  # Interpolated points
INTERP_D = 50

# Create original points
numbers = np.linspace(1, 1000, N, dtype=int)
d_values = np.linspace(2, 20, D, dtype=int)

# Create finer grid for interpolation
numbers_fine = np.linspace(1, 1000, INTERP_N)
d_values_fine = np.linspace(2, 20, INTERP_D)

# Pre-calculate all final values
final_steps = np.zeros((N, D))
for i in range(N):
    for j in range(D):
        _, steps = appx_squaring(numbers[i], d_values[j])
        final_steps[i, j] = max(0, steps)

# Create points for interpolation
points = np.array([(x, y) for x in range(N) for y in range(D)])
values = final_steps.flatten()

# Create fine meshgrid for interpolation
X_fine, Y_fine = np.meshgrid(np.linspace(0, N-1, INTERP_N), 
                            np.linspace(0, D-1, INTERP_D))

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
    
    # Set labels with actual values
    ax.set_xlabel('Numerator')
    ax.set_ylabel('Denominator')
    ax.set_zlabel('Steps')
    ax.set_title('Steps Required for Different Numbers and d Values')
    
    # Set custom tick labels
    x_ticks = np.linspace(0, N-1, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(numbers[int(i)]):,}' for i in x_ticks])
    
    y_ticks = np.linspace(0, D-1, 5)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'd={int(d_values[int(i)])}' for i in y_ticks])
    
    # Set view angle
    ax.view_init(elev=20, azim=frame/2)
    
    # Print progress
    completed = np.sum(current_steps_fine >= final_steps_fine)
    total = INTERP_N * INTERP_D
    print(f"\rProgress: {completed}/{total} complete ({completed/total*100:.1f}%)", end="")
    
    # Add color bar if it doesn't exist
    if not hasattr(update, 'colorbar'):
        update.colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        update.colorbar.set_label('Steps')
    
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