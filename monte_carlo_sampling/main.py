import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
from matplotlib.animation import FuncAnimation
import matplotlib.style as style

# Set dark background style
plt.style.use('dark_background')

def mixture_gaussian(x):
    """Mixture of two Gaussians: 0.1*N(-5,1) + 0.9*N(5,1)"""
    return 0.1 * norm.pdf(x, loc=-5, scale=1) + 0.9 * norm.pdf(x, loc=5, scale=1)

def sample_mixture_gaussian(N):
    """Generate N samples from the mixture distribution"""
    choices = np.random.binomial(1, 0.9, N)
    samples = np.where(choices == 0,
                      np.random.normal(-5, 1, N),
                      np.random.normal(5, 1, N))
    return samples

# Set random seed for reproducibility
np.random.seed(42)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Setup the plot
x = np.linspace(-8, 8, 1000)
true_density = mixture_gaussian(x)

# Plot true density
line_true, = ax.plot(x, true_density, 'cyan', label='True Distribution', linewidth=2)
ax.set_xlim(-8, 8)
ax.set_ylim(0, 0.5)
ax.set_title('Monte Carlo Sampling Animation', color='white', pad=20)
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.grid(True, alpha=0.2)

# Initialize N values
N_values = np.logspace(1, 4, 50, dtype=int)

def update(frame):
    ax.clear()
    N = N_values[frame]
    
    # Plot true density
    ax.plot(x, true_density, 'cyan', label='True Distribution', linewidth=2)
    
    # Generate new samples and create histogram
    samples = sample_mixture_gaussian(N)
    ax.hist(samples, bins=50, density=True, alpha=0.6, color='magenta',
            label='Monte Carlo Approximation')
    
    # Set plot properties
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.5)
    ax.set_title(f'Monte Carlo Sampling (N={N})', color='white', pad=20)
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.2)
    ax.legend()

# Create animation
anim = FuncAnimation(
    fig, 
    update,
    frames=len(N_values),
    interval=100,
    repeat=False
)

# Adjust layout
plt.tight_layout()

# Save animation
anim.save('monte_carlo_animation.gif', writer='pillow', fps=10, dpi=100)
plt.close()
