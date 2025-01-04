import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Tuple, Optional
import os

class HeatEquation2D:
    def __init__(
        self,
        nx: int = 50,
        ny: int = 50,
        dx: float = 0.1,
        dy: float = 0.1,
        alpha: float = 1.0,
        dt: float = 0.005
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.alpha = alpha
        self.dt = dt
        
        self.u = np.zeros((ny, nx))
        
        self.rx = alpha * dt / (dx * dx)
        self.ry = alpha * dt / (dy * dy)
        
        if self.rx + self.ry > 0.5:
            print("Warning: Solution may be unstable! Decrease dt or increase dx/dy")
    
    def set_initial_condition(self, func):
        x = np.linspace(0, self.nx * self.dx, self.nx)
        y = np.linspace(0, self.ny * self.dy, self.ny)
        X, Y = np.meshgrid(x, y)
        self.u = func(X, Y)
    
    def set_boundary_conditions(self, top=0, bottom=0, left=0, right=0):
        self.u[0, :] = bottom
        self.u[-1, :] = top
        self.u[:, 0] = left
        self.u[:, -1] = right
    
    def step(self) -> None:
        u_new = np.zeros_like(self.u)
        
        u_new[1:-1, 1:-1] = self.u[1:-1, 1:-1] + \
            self.rx * (self.u[1:-1, 2:] - 2*self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) + \
            self.ry * (self.u[2:, 1:-1] - 2*self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])
        
        u_new[0, :] = self.u[0, :]
        u_new[-1, :] = self.u[-1, :]
        u_new[:, 0] = self.u[:, 0]
        u_new[:, -1] = self.u[:, -1]
        
        self.u = u_new
    
    def solve(self, num_steps: int, save_video: bool = True, video_filename: str = "heat_equation_2d.mp4", 
             fps: int = 30, frame_skip: int = 2) -> Optional[str]:
        if not save_video:
            for _ in range(num_steps):
                self.step()
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        mesh = ax.imshow(self.u, cmap='hot', animated=True)
        plt.colorbar(mesh)
        ax.set_title("2D Heat Equation Simulation")
        
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        
        with writer.saving(fig, video_filename, dpi=100):
            for i in range(num_steps):
                self.step()
                if i % frame_skip == 0:
                    mesh.set_array(self.u)
                    writer.grab_frame()
        
        plt.close()
        return os.path.abspath(video_filename)

def example_usage():
    solver = HeatEquation2D(nx=100, ny=100, dx=0.1, dy=0.1, dt=0.005, alpha=2.0)
    
    def initial_temp(X, Y):
        pulse1 = np.exp(-((X-2.0)**2 + (Y-2.0)**2))
        pulse2 = np.exp(-((X-4.0)**2 + (Y-4.0)**2))
        return pulse1 + pulse2
    
    solver.set_initial_condition(initial_temp)
    
    solver.set_boundary_conditions(top=0, bottom=0, left=0, right=0)
    
    video_path = solver.solve(num_steps=200, fps=30, frame_skip=2)
    print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    example_usage() 