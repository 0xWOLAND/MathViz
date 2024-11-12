import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def appx_squaring(n, d):
    steps = 0
    try:
        while n % d:
            next_n = n * ((n + d + 1) // d)
            if next_n > sys.maxsize:
                return (-1, -1)
            n = next_n
            steps += 1
        return (n, steps)
    except OverflowError:
        return (-1, -1)

# Initialize data
N = 50  # Number of x values
D = 5   # Number of different d values
numbers = sorted(random.sample(range(1, 10**6), N))
d_values = sorted(random.sample(range(2, 20), D))
current_steps = np.zeros((N, D))
final_steps = np.full((N, D), -1)
processing = np.ones((N, D), dtype=bool)
started = np.zeros((N, D), dtype=bool)
lock = threading.Lock()

print(f"Calculating steps for {N} numbers with {D} different d values...")
print(f"Numbers range: {min(numbers):,} to {max(numbers):,}")
print(f"d values: {d_values}")

# Create figure and axis
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create the base for the bars
x_pos = np.arange(N)
y_pos = np.arange(D)
X, Y = np.meshgrid(x_pos, y_pos)
X, Y = X.ravel(), Y.ravel()
dx = dy = 0.8

# Calculate final steps in parallel
def calculate_final_steps(idx):
    i, j = idx // D, idx % D
    number = numbers[i]
    d = d_values[j]
    with lock:
        started[i, j] = True
    _, steps = appx_squaring(number, d)
    with lock:
        final_steps[i, j] = steps
    return idx, steps

# Start background calculation
total_calcs = N * D
with ThreadPoolExecutor(max_workers=min(32, total_calcs)) as executor:
    futures = [executor.submit(calculate_final_steps, i) for i in range(total_calcs)]

def update(frame):
    # Remove previous bars
    ax.cla()
    
    # Update steps
    for i in range(N):
        for j in range(D):
            if started[i, j] and processing[i, j]:
                if final_steps[i, j] != -1:
                    if current_steps[i, j] < final_steps[i, j]:
                        current_steps[i, j] += 1
                    else:
                        processing[i, j] = False

    # Create the bars
    Z = current_steps.ravel()
    colors = []
    
    for idx in range(N*D):
        i, j = idx // D, idx % D
        if not started[i, j]:
            colors.append('gray')
        elif not processing[i, j]:
            colors.append('green')
        else:
            colors.append('blue')

    # Plot the bars
    bars = ax.bar3d(X, Y, np.zeros_like(Z), dx, dy, Z, color=colors, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('Number Index')
    ax.set_ylabel('d Value')
    ax.set_zlabel('Steps')
    ax.set_title('Steps Required for Different Numbers and d Values')
    
    # Update axis limits
    ax.set_xlim(-1, N)
    ax.set_ylim(-1, D)
    max_height = max(max(Z), 1)
    ax.set_zlim(0, max_height + 1)
    
    # Update labels
    ax.set_xticks(np.arange(0, N, N//5))
    ax.set_xticklabels([f"{numbers[i]:,}" for i in range(0, N, N//5)], rotation=45)
    ax.set_yticks(np.arange(D))
    ax.set_yticklabels([f"d={d}" for d in d_values])
    
    # Print progress
    completed = np.sum(~processing)
    total = N * D
    print(f"\rProgress: {completed}/{total} calculations complete ({completed/total*100:.1f}%)", end="")
    
    # Stop animation if all numbers are done
    if not any(processing.ravel()):
        anim.event_source.stop()
        print("\n\nFinal Statistics:")
        valid_steps = final_steps[final_steps != -1]
        if len(valid_steps) > 0:
            print(f"Average steps: {np.mean(valid_steps):.2f}")
            print(f"Maximum steps: {np.max(valid_steps)}")
            print(f"Minimum steps: {np.min(valid_steps)}")
            print(f"Overflow count: {np.sum(final_steps == -1)}")
            
            # Find best d value
            avg_steps_per_d = np.array([np.mean(final_steps[:, j][final_steps[:, j] != -1]) 
                                      for j in range(D)])
            best_d_idx = np.argmin(avg_steps_per_d)
            print(f"\nBest d value: {d_values[best_d_idx]} "
                  f"(avg steps: {avg_steps_per_d[best_d_idx]:.2f})")
    
    # Rotate view for better visualization
    ax.view_init(elev=20, azim=frame % 360)
    
    return bars,

# Create animation
anim = animation.FuncAnimation(
    fig, 
    update,
    frames=None,
    interval=100,  # Slower updates for better visibility
    blit=False    # Set to False to ensure proper updates
)

plt.tight_layout()
plt.show()