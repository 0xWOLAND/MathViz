import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def quadratic_function(x, A, b, c=0):
    return 0.5 * x.T @ A @ x - b.T @ x + c

def conjugate_gradient(A, b, x0, tol=1e-8):
    dim = A.shape[0]
    x = x0
    r = b - A @ x
    d = r.copy()
    
    xs = [x.copy()]
    ds = [d.copy()]
    
    rTr_old = r.T @ r
    
    for _ in range(dim):
        Ad = A @ d
        alpha = rTr_old / (d.T @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rTr_new = r.T @ r
        beta = rTr_new / rTr_old
        d = r + beta * d
        
        xs.append(x.copy())
        ds.append(d.copy())
        
        rTr_old = rTr_new
        
        if np.sqrt(rTr_new) < tol:
            break
    
    return np.array(xs), np.array(ds)

def interpolate_path(xs, points_per_segment=50):
    """Create a path that follows exact line segments between optimization points"""
    path_points = []
    
    for i in range(len(xs) - 1):
        t = np.linspace(0, 1, points_per_segment)
        segment = np.outer(1-t, xs[i]) + np.outer(t, xs[i+1])
        path_points.append(segment)
    
    return np.vstack(path_points)


def create_and_save_visualization(A, b, xs, ds, filename='cgd_visualization.mp4', fps=30):
    """
    Creates visualization and saves it as an MP4 file
    
    Parameters:
        A, b, xs, rs, ds: Algorithm data
        filename: Output MP4 filename
        fps: Frames per second for the video
    """
    plt.style.use('dark_background')
    
    # Create interpolated path
    xs_smooth = interpolate_path(xs, points_per_segment=50)
    total_steps = len(xs_smooth)
    
    # Compute optimal point
    x_opt = np.linalg.solve(A, b)
    
    # Set up the figure with higher DPI for better video quality
    fig = plt.figure(figsize=(15, 6), dpi=150)
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Determine plot bounds
    path_min = np.minimum(xs_smooth.min(axis=0), x_opt)
    path_max = np.maximum(xs_smooth.max(axis=0), x_opt)
    path_center = (path_min + path_max) / 2
    path_radius = np.max(np.abs(path_max - path_min)) / 2
    
    # Create grid for plotting
    x1 = np.linspace(path_center[0] - path_radius, path_center[0] + path_radius, 150)
    x2 = np.linspace(path_center[1] - path_radius, path_center[1] + path_radius, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = np.array([X1[i,j], X2[i,j]])
            Z[i,j] = quadratic_function(x, A, b)
    
    # 3D surface plot
    surf = ax1.plot_surface(X1, X2, Z, cmap='magma', alpha=0.8)
    ax1.set_title('Optimization Path (3D)', color='white', pad=20, fontsize=12)
    ax1.set_xlabel('x₁', color='white', labelpad=10)
    ax1.set_ylabel('x₂', color='white', labelpad=10)
    ax1.set_zlabel('f(x)', color='white', labelpad=10)
    ax1.tick_params(colors='white')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(True, alpha=0.2)
    
    # Contour plot
    margin = 0.5
    ax2.set_xlim(path_center[0] - path_radius - margin, 
                 path_center[0] + path_radius + margin)
    ax2.set_ylim(path_center[1] - path_radius - margin, 
                 path_center[1] + path_radius + margin)
    
    contours = ax2.contour(X1, X2, Z, levels=20, cmap='magma')
    ax2.set_title('Conjugate Directions', color='white', pad=20, fontsize=12)
    ax2.set_xlabel('x₁', color='white')
    ax2.set_ylabel('x₂', color='white')
    ax2.grid(True, alpha=0.2)
    
    # Initialize animation elements
    point3d, = ax1.plot([], [], [], 'wo', markersize=8, markeredgecolor='cyan', 
                       markerfacecolor='cyan', alpha=0.8)
    path3d, = ax1.plot([], [], [], '-', color='cyan', linewidth=2, alpha=0.6)
    point2d, = ax2.plot([], [], 'wo', markersize=8, markeredgecolor='cyan',
                       markerfacecolor='cyan', alpha=0.8)
    path2d, = ax2.plot([], [], '-', color='cyan', linewidth=2, alpha=0.6)
    
    # Add original points
    ax2.plot(xs[:, 0], xs[:, 1], 'wo', markersize=5, alpha=0.7, label='Iterations')
    
    # Add optimal point
    opt_z = quadratic_function(x_opt, A, b)
    ax1.plot([x_opt[0]], [x_opt[1]], [opt_z], 
             marker='*', color='lime', markersize=12, label='Optimum')
    ax2.plot([x_opt[0]], [x_opt[1]], marker='*', color='lime', 
             markersize=12, label='Optimum')
    
    # Plot conjugate directions
    for i in range(min(len(ds)-1, 2)):
        start = xs[i]
        direction = ds[i] / np.linalg.norm(ds[i])
        scale = path_radius * 0.8
        end = start + scale * direction
        ax2.arrow(start[0], start[1], scale*direction[0], scale*direction[1],
                 color='red' if i == 0 else 'yellow', width=path_radius*0.02,
                 head_width=path_radius*0.08, head_length=path_radius*0.1,
                 alpha=0.7, label=f'd_{i}')
    
    ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    
    # Animation settings
    n_rotations = 1  # Number of complete rotations
    frames_per_rotation = 90   # Number of frames per rotation
    n_frames = n_rotations * frames_per_rotation
    
    def update(frame):
        # Calculate the phase of the optimization (0 to 1)
        opt_phase = min(frame / frames_per_rotation, 1.0)
        path_frame = int(opt_phase * total_steps)
        path_frame = min(path_frame, total_steps - 1)
        
        # Update 3D point and path
        current_x = xs_smooth[path_frame]
        current_z = quadratic_function(current_x, A, b)
        point3d.set_data([current_x[0]], [current_x[1]])
        point3d.set_3d_properties([current_z])
        
        path3d.set_data(xs_smooth[:path_frame+1, 0], xs_smooth[:path_frame+1, 1])
        path3d.set_3d_properties([quadratic_function(x, A, b) for x in xs_smooth[:path_frame+1]])
        
        # Update 2D point and path
        point2d.set_data([current_x[0]], [current_x[1]])
        path2d.set_data(xs_smooth[:path_frame+1, 0], xs_smooth[:path_frame+1, 1])
        
        # Continuous rotation (2 degrees per frame)
        azimuth = frame * 2 % 360
        ax1.view_init(elev=20, azim=azimuth)
        
        return point3d, path3d, point2d, path2d
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Set up the writer
    writer = FFMpegWriter(
        fps=fps, 
        metadata=dict(artist='CGD Visualization'),
        bitrate=5000  # Higher bitrate for better quality
    )
    
    # Save the animation
    anim.save(filename, writer=writer)
    plt.close()
    plt.show()
    
    print(f"Animation saved as {filename}")

if __name__ == "__main__":
    # Create test problem from the paper
    A = np.array([[4, 1], 
                  [1, 2]])
    b = np.array([0, 2])
    x0 = np.array([3.0, 3.0])
    
    # Run conjugate gradient
    xs, ds = conjugate_gradient(A, b, x0)
    
    # Create and save visualization
    create_and_save_visualization(A, b, xs, ds, 
                                filename='cgd_visualization.mp4',
                                fps=30)