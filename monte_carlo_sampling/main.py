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

def importance_sampling(N):
    """
    Importance sampling using a N(0,5) proposal distribution
    Returns samples and their weights
    """
    # Proposal distribution: N(0,5)
    proposal_samples = np.random.normal(0, 5, N)
    
    # Compute unnormalized weights
    target_density = 0.1 * norm.pdf(proposal_samples, loc=-5, scale=1) + \
                    0.9 * norm.pdf(proposal_samples, loc=5, scale=1)
    proposal_density = norm.pdf(proposal_samples, loc=0, scale=5)
    unnormalized_weights = target_density / proposal_density
    
    # Normalize weights
    weights = unnormalized_weights / np.sum(unnormalized_weights)
    
    return proposal_samples, weights

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
ax.set_title('Monte Carlo vs Importance Sampling', color='white', pad=20)
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
    
    # Direct Monte Carlo sampling
    mc_samples = sample_mixture_gaussian(N)
    ax.hist(mc_samples, bins=50, density=True, alpha=0.4, color='magenta',
            label='Monte Carlo', histtype='stepfilled')
    
    # Importance sampling
    is_samples, is_weights = importance_sampling(N)
    ax.hist(is_samples, bins=50, density=True, alpha=0.4, color='yellow',
            weights=is_weights * N, label='Importance Sampling', histtype='stepfilled')
    
    # Plot proposal distribution (scaled for visibility)
    proposal = norm.pdf(x, loc=0, scale=5) * 0.5
    ax.plot(x, proposal, '--', color='green', alpha=0.5, label='Proposal Distribution')
    
    # Set plot properties
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.5)
    ax.set_title(f'Monte Carlo vs Importance Sampling (N={N})', color='white', pad=20)
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
anim.save('sampling_comparison.gif', writer='pillow', fps=10, dpi=100)
plt.close()
