import numpy as np
import matplotlib.pyplot as plt
import imageio

class Particle:
    def __init__(self, func):
        self.position = None
        self.velocity = None
        self.best_position = None
        self.best_eval = func(self.position)

    def init_particle(self, min_range: list[int], max_range: list[int]):
        self.position = np.random.uniform(min_range, max_range)
        self.velocity = np.random.uniform(-1, 1, size=len(min_range)) * (max_range - min_range)
        self.best_position = np.copy(self.position)

    def update_velocity(self, omega, fip, fig, global_best_position):
        cognitive = fip * np.random.rand() * (self.best_position - self.position)
        social = fig * np.random.rand() * (global_best_position - self.position)
        self.velocity = omega * self.velocity + cognitive + social

    def update_position(self, min_range, max_range):
        self.position += self.velocity
        self.position = np.clip(self.position, min_range, max_range)

    def evaluate(self, func):
        current_eval = func(self.position)
        if current_eval < self.best_eval:
            self.best_position = np.copy(self.position)
            self.best_eval = current_eval


class ParticleSwarmOptimizer:
    def __init__(self, func, min_range, max_range, particles_no: int, iterations: int, omega, fip, fig):
        self.func = func
        self.min_range = np.array(min_range)
        self.max_range = np.array(max_range)
        self.particles_no = particles_no
        self.iterations = iterations
        self.omega = omega
        self.fip = fip
        self.fig = fig

        self.particles = [Particle.init_particle(func, self.min_range, self.max_range) for _ in range(particles_no)]
        self.global_best_position = None
        self.global_best_eval = float('inf')
        self.update_global_best()

    def update_global_best(self):
        for particle in self.particles:
            if particle.best_eval < self.global_best_eval:
                self.global_best_eval = particle.best_eval
                self.global_best_position = np.copy(particle.best_position)

    def optimize(self):
        frames = []

        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")

            for particle in self.particles:
                particle.update_velocity(self.omega, self.fip, self.fig, self.global_best_position)
                particle.update_position(self.min_range, self.max_range)
                particle.evaluate(self.func)

            self.update_global_best()
            frame = self.plot_particles(iteration)
            frames.append(frame)

        self.save_gif(frames, "particleswarm.gif")
        return self.global_best_position, self.global_best_eval

    def plot_particles(self, iteration):
        plt.figure()
        for particle in self.particles:
            plt.plot(particle.position[0] * 1e6, particle.position[1], 'x')

        plt.xlim(self.min_range[0] * 1e6, self.max_range[0] * 1e6)
        plt.ylim(self.min_range[1], self.max_range[1])
        plt.xlabel('thickness [µm]')
        plt.ylabel('refractive index')
        plt.title(f"Iteration {iteration + 1}")
        plt.grid()

        # Save frame for GIF
        plt.savefig("temp_frame.png")
        plt.close()
        return imageio.imread("temp_frame.png")

    def save_gif(self, frames, filename):
        imageio.mimsave(filename, frames, format="GIF", duration=1)


# Example usage
def example_function(position):
    return np.sum(position ** 2)  # Minimize the sum of squares

if __name__ == "__main__":
    optimizer = ParticleSwarmOptimizer(
        func=example_function,
        min_range=[0, 0],
        max_range=[10, 10],
        particles_no=100,
        iterations=50,
        omega=0.5,
        fip=1.5,
        fig=1.5
    )

    best_position, best_eval = optimizer.optimize()
    print(f"Best Position: {best_position}, Best Evaluation: {best_eval}")
