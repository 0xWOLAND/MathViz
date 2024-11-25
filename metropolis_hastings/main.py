import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def rosenbrock(x, y, a=1, b=100):
    """
    Rosenbrock function (banana function)
    f(x,y) = (a-x)^2 + b(y-x^2)^2
    """
    return (a - x)**2 + b * (y - x**2)**2

def proposal(current_state, step_size=0.3):
    """Generate proposal state using random walk"""
    return current_state + np.random.normal(0, step_size, size=2)

def metropolis_hastings(n_iterations=10000, step_size=0.3):
    # Initial state
    current_state = np.array([0.0, 0.0])
    
    # Storage for chain
    chain = np.zeros((n_iterations, 2))
    
    # Store initial state
    chain[0] = current_state
    
    # Run Metropolis-Hastings
    for i in range(1, n_iterations):
        # Generate proposal
        proposed_state = proposal(current_state, step_size)
        
        # Calculate acceptance ratio
        current_likelihood = -rosenbrock(current_state[0], current_state[1])
        proposed_likelihood = -rosenbrock(proposed_state[0], proposed_state[1])
        
        # Log acceptance ratio
        log_ratio = proposed_likelihood - current_likelihood
        
        # Accept or reject
        if np.log(np.random.random()) < log_ratio:
            current_state = proposed_state
            
        chain[i] = current_state
    
    return chain

def create_3d_animation(chain, interval=20, n_frames=400):
    """Create 3D animation of the sampling process"""
    # Set up the figure and 3D axis with black background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    
    # Create main 3D axis
    ax = fig.add_subplot(111, projection='3d')
    
    # Set figure and axis background color
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Fixed view limits
    view_limits = {
        'x': [-2.5, 2.5],
        'y': [-1.5, 3.5],
        'z': [0, 1500]
    }
    
    # Create surface plot of Rosenbrock function
    x = np.linspace(view_limits['x'][0], view_limits['x'][1], 200)
    y = np.linspace(view_limits['y'][0], view_limits['y'][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    # Plot the surface with transparency
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3,
                            rstride=2, cstride=2, linewidth=0,
                            antialiased=True)
    
    # Initialize empty line collection
    lines = []
    
    # Set labels and title with white color and LaTeX formula
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    title = (r'Metropolis-Hastings Sampling of Rosenbrock Function'
             '\n' r'$f(x,y) = (1-x)^2 + 100(y-x^2)^2$')
    ax.set_title(title, color='white', pad=20)
    
    # Set white color for axis ticks
    ax.tick_params(colors='white')
    
    # Set fixed view limits
    ax.set_xlim(view_limits['x'])
    ax.set_ylim(view_limits['y'])
    ax.set_zlim(view_limits['z'])
    
    def init():
        return []
    
    def update(frame):
        # Remove old lines
        for line in lines:
            line.remove()
        lines.clear()
        
        # Fixed camera angle with slight elevation
        ax.view_init(elev=30, azim=45)
        
        # Get points up to current frame
        idx = int((frame / n_frames) * len(chain))
        points = chain[:idx]
        
        if len(points) > 1:
            # Create segments for coloring
            segments = 50
            segment_size = max(len(points) // segments, 1)
            
            for i in range(0, len(points)-segment_size, segment_size):
                segment = points[i:i+segment_size+1]
                x_data = segment[:, 0]
                y_data = segment[:, 1]
                z_data = rosenbrock(x_data, y_data)
                
                # Create color gradient from dark red to bright red
                progress = i / len(points)
                color = plt.cm.Reds(0.3 + 0.7 * progress)
                
                # Add new line segment
                line, = ax.plot(x_data, y_data, z_data, 
                              color=color, linewidth=1.5, alpha=0.8)
                lines.append(line)
        
        return lines
    
    # Create and display animation
    anim = FuncAnimation(fig, update, frames=n_frames, 
                        init_func=init, interval=interval,
                        blit=True)
    
    # Remove grid for cleaner look
    ax.grid(False)
    
    # Set axis pane colors to dark
    ax.xaxis.set_pane_color((0.02, 0.02, 0.02, 1.0))
    ax.yaxis.set_pane_color((0.02, 0.02, 0.02, 1.0))
    ax.zaxis.set_pane_color((0.02, 0.02, 0.02, 1.0))
    
    plt.show()

if __name__ == "__main__":
    # Run simulation
    chain = metropolis_hastings(n_iterations=100000, step_size=0.1)
    
    # Create animation
    create_3d_animation(chain, interval=30, n_frames=400)  # slightly slower interval for smoother color transition 