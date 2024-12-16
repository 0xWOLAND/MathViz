import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

def weierstrass_term(x, n, a=0.5, b=7):
    """Calculate the nth term of the Weierstrass function."""
    return a**n * np.cos(b**n * np.pi * x)

class WeierstrassAnimation:
    def __init__(self, initial_range=(-2, 2), final_range=(-0.25, 0.25), num_points=1000, num_terms=15, a=0.5, b=7):
        # Set up the figure with dark theme
        plt.style.use('dark_background')
        # Reduced figure size and DPI for smaller file size
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=72)
        self.fig.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')
        
        # Store parameters
        self.initial_range = initial_range
        self.final_range = final_range
        self.num_points = num_points  # Reduced number of points
        self.num_terms = num_terms
        self.a = a
        self.b = b
        
        # Create initial x values
        self.x = np.linspace(initial_range[0], initial_range[1], num_points)
        
        # Reduced number of trails for optimization
        self.lines = []
        num_trails = 10  # Reduced from 20
        for _ in range(num_trails):
            line, = self.ax.plot([], [], lw=1.5, alpha=0)  # Reduced line width
            self.lines.append(line)
        
        # Set initial axis properties
        self.ax.set_xlim(initial_range)
        self.ax.set_ylim(-2, 2)
        self.ax.grid(True, alpha=0.2, color='#333333')
        
        # Simplified title with smaller font
        title = (r"Weierstrass Function: $f(x) = \sum a^n \cos(b^n\pi x)$" "\n" 
                f"a = {a}, b = {b}")
        self.ax.set_title(title, color='#F5F5F5', pad=10, fontsize=12)
        
        self.ax.set_xlabel('x', color='#F5F5F5', fontsize=10)
        self.ax.set_ylabel('f(x)', color='#F5F5F5', fontsize=10)
        
        # Style the axes
        for spine in self.ax.spines.values():
            spine.set_color('#444444')
        self.ax.tick_params(colors='#F5F5F5', labelsize=8)
        
        # Add text for term counter and zoom level
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,
                                    color='#F5F5F5', fontsize=10)
        
        self.main_color = '#F5F5F5'
        
        # Reduced number of frames
        self.total_frames = num_terms * 2  # Reduced from 4 to 2

    def get_zoom_range(self, progress):
        """Calculate current zoom range based on animation progress"""
        zoom_factor = np.exp(progress * np.log(self.initial_range[1]/self.final_range[1]))
        current_half_width = self.initial_range[1] / zoom_factor
        drift = 0.3 * np.sin(progress * 2 * np.pi) * current_half_width
        return (-current_half_width + drift, current_half_width + drift)

    def init(self):
        for line in self.lines:
            line.set_data([], [])
        self.info_text.set_text('')
        return self.lines + [self.info_text]

    def animate(self, frame):
        progress = frame / self.total_frames
        actual_frame = frame / 2  # Adjusted for reduced frame count
        integer_frame = int(actual_frame)
        frac = actual_frame - integer_frame
        
        current_range = self.get_zoom_range(progress)
        self.x = np.linspace(current_range[0], current_range[1], self.num_points)
        self.ax.set_xlim(current_range)
        
        current_y = np.zeros_like(self.x)
        next_y = np.zeros_like(self.x)
        
        for n in range(integer_frame + 1):
            current_y += weierstrass_term(self.x, n, self.a, self.b)
        
        if integer_frame + 1 < self.num_terms:
            next_y = current_y + weierstrass_term(self.x, integer_frame + 1, self.a, self.b)
            y = current_y * (1 - frac) + next_y * frac
        else:
            y = current_y
        
        for i, line in enumerate(self.lines):
            alpha = 1.0 if i == 0 else (1.0 - i/len(self.lines)) * 0.3
            line.set_data(self.x, y)
            line.set_color(self.main_color)
            line.set_alpha(alpha)
        
        zoom_level = self.initial_range[1] / (current_range[1] - current_range[0]) * 2
        self.info_text.set_text(f'Terms: {min(actual_frame, self.num_terms):.1f}\nZoom: {zoom_level:.1f}x')
        
        return self.lines + [self.info_text]


    def create_animation(self, save_gif=True):
        anim = FuncAnimation(
            self.fig, self.animate, frames=self.total_frames,
            init_func=self.init, blit=True, interval=40
        )
        
        if save_gif:
            if not os.path.exists('animations'):
                os.makedirs('animations')
            
            # Corrected PillowWriter settings
            writer = PillowWriter(
                fps=15,  # Reduced framerate for smaller file
                metadata=dict(artist='Me')
            )
            filename = 'animations/weierstrass_fractal_optimized.gif'
            
            print(f"Saving optimized animation to {filename}...")
            # Save with tight layout to remove excess margins
            plt.tight_layout()
            anim.save(filename, writer=writer)
            print("Animation saved successfully!")
        
        return anim

# Create and display the animation
if __name__ == "__main__":
    animation = WeierstrassAnimation()
    anim = animation.create_animation(save_gif=True)
    plt.show()