import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Set the style for dark background
plt.style.use('dark_background')

# Define the function and its derivative
def f(x):
    return x**2 - 30

def df(x):
    return 2*x

# Newton's method implementation
def newton_step(x):
    return x - f(x)/df(x)

# Add convergence check
def has_converged(x, threshold=0.001):  # Decreased threshold from 0.1 to 0.001
    return abs(f(x)) < threshold

# Create figure and axis with lower DPI
fig, ax = plt.subplots(figsize=(12, 8), dpi=50)  # Reduced DPI from default ~100 to 50
ax.set_title("Newton's Method for sqrt(30)")

# Set up the plot range
x = np.linspace(-5, 15, 1000)
y = f(x)

# Plot the function
line, = ax.plot(x, y, 'cyan', label='f(x) = x² - 30')

# Add grid and axes
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.1)

# Starting point
x_start = 10.0
points = []
point, = ax.plot([], [], 'ro', markersize=10)
path, = ax.plot([], [], 'r-', alpha=0.5)  # Add a line to track the path
tangent_line, = ax.plot([], [], 'yellow', alpha=0.5)

# Add text for current value
text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
               color='white', fontsize=10, verticalalignment='top')

# Set axis limits for main plot
ax.set_xlim(-5, 15)
ax.set_ylim(-100, 1500)

# Add legend
ax.legend()

def animate(frame):
    if frame == 0:
        x_current = x_start
        points.clear()  # Clear points at start of each loop
    else:
        if not points:  # If points is empty (happens after convergence)
            x_current = x_start
        else:
            x_current = points[-1]
            if has_converged(x_current):  # If converged, start over
                points.clear()
                x_current = x_start
    
    # Calculate next point using Newton's method
    x_next = newton_step(x_current)
    points.append(x_next)
    
    # Plot current point and path
    point.set_data([x_current], [f(x_current)])
    
    # Update path
    x_path = [x_start] + points
    y_path = [f(x) for x in x_path]
    path.set_data(x_path, y_path)
    
    # Plot tangent line
    slope = df(x_current)
    x_tangent = np.array([x_current - 0.5, x_current + 0.5])
    y_tangent = f(x_current) + slope * (x_tangent - x_current)
    tangent_line.set_data(x_tangent, y_tangent)
    
    # Update text showing current value and convergence status
    status = "CONVERGED!" if has_converged(x_current) else "searching..."
    text.set_text(f'Current x: {x_current:.8f}\nf(x): {f(x_current):.8f}\n{status}')
    
    return point, tangent_line, path, text

# Create animation with more frames and shorter interval
anim = FuncAnimation(fig, animate, frames=20,
                    interval=500,
                    blit=True, repeat=True)

# Save as GIF with reduced quality
writer = PillowWriter(
    fps=2,
    metadata=dict(artist='Me'),
    bitrate=1000,  # Lower bitrate
)
anim.save('newton_method.gif', writer=writer, dpi=50)  # Save with lower DPI
plt.close()
