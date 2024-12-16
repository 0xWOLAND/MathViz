import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
import matplotlib as mpl

class PAdicNumber:
    def __init__(self, digits: List[int], p: int):
        self.digits = digits  # coefficients in p-adic expansion
        self.p = p           # prime base
        
    def __str__(self):
        return f"...{self.digits[::-1]}"
    
def visualize_padic_animated(number: PAdicNumber, max_radius: int = 5, duration_sec: int = 10):
    """
    Create an animated visualization of a p-adic number with pulsing circles
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    
    # Set up the plot
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Create custom colormap with better visibility on black background
    colors = plt.cm.plasma(np.linspace(0, 1, number.p))
    
    # Store circles for animation
    circles = []
    texts = []
    
    # Create initial circles
    for i, digit in enumerate(number.digits):
        radius = max_radius * (1 - 1/(i + 2))
        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            color=colors[digit],
            alpha=0.6,
            linewidth=2
        )
        ax.add_patch(circle)
        circles.append(circle)
        
        # Add text with white color
        angle = np.pi/4
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        text = ax.text(x, y, f"{digit}×{number.p}^{i}", 
                      fontsize=10, ha='center', va='center',
                      color='white')
        texts.append(text)
    
    title = plt.title(f"p-adic number visualization (p={number.p})",
                     color='white', pad=20)
    
    def animate(frame):
        t = frame / 50  # Slow down the animation
        
        for i, (circle, text) in enumerate(zip(circles, texts)):
            # Pulse the circles
            base_radius = max_radius * (1 - 1/(i + 2))
            pulse = 0.05 * np.sin(2 * np.pi * t - i * 0.5)
            circle.set_radius(base_radius + pulse)
            
            # Make the circles glow
            alpha = 0.6 + 0.4 * np.sin(2 * np.pi * t - i * 0.5) ** 2
            circle.set_alpha(alpha)
            
            # Update text positions
            angle = np.pi/4
            x = (base_radius + pulse) * np.cos(angle)
            y = (base_radius + pulse) * np.sin(angle)
            text.set_position((x, y))
        
        return circles + texts + [title]
    
    frames = int(50 * duration_sec)  # 50 frames per second
    anim = FuncAnimation(fig, animate, frames=frames, interval=20, blit=True)
    
    return fig, ax, anim

def main():
    # Set global dark theme
    plt.style.use('dark_background')
    
    # Example: Visualize -1 in 5-adic expansion
    p = 5
    digits = [4] * 8  # Taking first 8 terms of the expansion
    number = PAdicNumber(digits, p)
    
    print(f"Visualizing {number} (base {p})")
    fig, ax, anim = visualize_padic_animated(number, duration_sec=20)
    plt.show()
    
    # Example: Visualize 1/3 in 7-adic expansion
    p = 7
    digits = [5, 5, 5, 5, 5, 5, 5, 5]  # Taking first 8 terms
    number = PAdicNumber(digits, p)
    
    print(f"Visualizing {number} (base {p})")
    fig, ax, anim = visualize_padic_animated(number, duration_sec=20)
    plt.show()

if __name__ == "__main__":
    main()
