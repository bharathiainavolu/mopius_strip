
import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R, w, n):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w/2, w/2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self.generate_coordinates()

    def generate_coordinates(self):
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def surface_area(self):
        def integrand(v, u):
            dx_du = -(self.R + v * np.cos(u / 2)) * np.sin(u) - 0.5 * v * np.sin(u / 2) * np.cos(u)
            dx_dv = np.cos(u / 2) * np.cos(u)
            dy_du = (self.R + v * np.cos(u / 2)) * np.cos(u) - 0.5 * v * np.sin(u / 2) * np.sin(u)
            dy_dv = np.cos(u / 2) * np.sin(u)
            dz_du = 0.5 * v * np.cos(u / 2)
            dz_dv = np.sin(u / 2)

            cross_x = dy_du * dz_dv - dz_du * dy_dv
            cross_y = dz_du * dx_dv - dx_du * dz_dv
            cross_z = dx_du * dy_dv - dy_du * dx_dv

            return np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        area, _ = dblquad(integrand, 0, 2 * np.pi, lambda u: -self.w/2, lambda u: self.w/2)
        return area

    def edge_length(self):
        u = self.u
        v = self.w / 2
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
        return length

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, color='cyan', edgecolor='k', alpha=0.7)
        plt.title('Mobius Strip')
        plt.show()

if __name__ == "__main__":
    strip = MobiusStrip(R=1, w=0.4, n=100)
    strip.plot()
    print("Surface Area:", strip.surface_area())
    print("Edge Length:", strip.edge_length())
