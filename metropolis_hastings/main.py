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
    chain = np.zeros((n_iterations, 2))
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

def create_3d_animation(chain, interval=20, n_frames=400, save_path='metropolis_hastings.gif'):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    view_limits = {
        'x': [-2.5, 2.5],
        'y': [-1.5, 3.5],
        'z': [0, 1500]
    }
    
    x = np.linspace(view_limits['x'][0], view_limits['x'][1], 200)
    y = np.linspace(view_limits['y'][0], view_limits['y'][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3,
                            rstride=2, cstride=2, linewidth=0,
                            antialiased=True)
    
    lines = []
    
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    title = (r'Metropolis-Hastings Sampling of Rosenbrock Function'
             '\n' r'$f(x,y) = (1-x)^2 + 100(y-x^2)^2$')
    ax.set_title(title, color='white', pad=20)
    
    ax.tick_params(colors='white')
    
    ax.set_xlim(view_limits['x'])
    ax.set_ylim(view_limits['y'])
    ax.set_zlim(view_limits['z'])
    
    def init():
        return []
    
    def update(frame):
        for line in lines:
            line.remove()
        lines.clear()
        
        ax.view_init(elev=30, azim=45)
        
        idx = int((frame / n_frames) * len(chain))
        points = chain[:idx]
        
        if len(points) > 1:
            segments = 50
            segment_size = max(len(points) // segments, 1)
            
            for i in range(0, len(points)-segment_size, segment_size):
                segment = points[i:i+segment_size+1]
                x_data = segment[:, 0]
                y_data = segment[:, 1]
                z_data = rosenbrock(x_data, y_data)
                
                progress = i / len(points)
                color = plt.cm.Reds(0.3 + 0.7 * progress)
                
                line, = ax.plot(x_data, y_data, z_data, 
                              color=color, linewidth=1.5, alpha=0.8)
                lines.append(line)
        
        return lines
    
    anim = FuncAnimation(fig, update, frames=n_frames, 
                        init_func=init, interval=interval,
                        blit=True)
    
    ax.grid(False)
    
    ax.xaxis.set_pane_color((0.02, 0.02, 0.02, 1.0))
    ax.yaxis.set_pane_color((0.02, 0.02, 0.02, 1.0))
    ax.zaxis.set_pane_color((0.02, 0.02, 0.02, 1.0))
    
    anim.save(save_path, writer='pillow', fps=1000/interval)
    
    plt.show()

if __name__ == "__main__":
    chain = metropolis_hastings(n_iterations=100000, step_size=0.1)
    create_3d_animation(chain, interval=30, n_frames=400, 
                       save_path='metropolis_hastings.gif')