import numpy as np
import matplotlib.pyplot as plt

class RandomWalk1D:
    def __init__(self, p=0.5, repetition=100, steps=1000):
        self.p = p
        self.position = 0
        self.repetition = repetition
        self.steps = steps
        self.random_rand = True
        self.random_normal = False
        self.random_exp = False

    def step_rand(self):
        if np.random.rand() < self.p:
            self.position += 1
        else:
            self.position -= 1

    def step_normal(self):
        if np.random.rand() < self.p:
            self.position += 1
        else:
            self.position -= 1

    def step_exp(self):
        if np.random.rand() < self.p:
            self.position += 1
        else:
            self.position -= 1

    def walk(self, n_steps):
        self.position = 0
        positions = [self.position]
        for _ in range(n_steps):
            if self.random_rand:
                self.step_rand()
            elif self.random_normal:
                self.step_normal()
            elif self.random_exp:
                self.step_exp()
            positions.append(self.position)
        return np.array(positions)

    def usefunction(self):
        fig, axs = plt.subplots(6, 1, figsize=(10, 30))

        # Uniform Distribution
        self.random_rand = True
        self.random_normal = False
        self.random_exp = False
        uniform_positions = np.zeros((self.repetition, self.steps + 1))
        for i in range(self.repetition):
            positions = self.walk(self.steps)
            uniform_positions[i, :] = positions
            axs[0].plot(positions)
        std_uniform = np.std(uniform_positions**2, axis=0)
        axs[1].plot(std_uniform)
        axs[0].set_title('Random Walk with Uniform Distribution')
        axs[1].set_title('Standard Deviation of Squared Positions with Uniform Distribution')

        # Normal Distribution
        self.random_rand = False
        self.random_normal = True
        self.random_exp = False
        normal_positions = np.zeros((self.repetition, self.steps + 1))
        for i in range(self.repetition):
            positions = self.walk(self.steps)
            normal_positions[i, :] = positions
            axs[2].plot(positions)
        std_normal = np.std(normal_positions**2, axis=0)
        axs[3].plot(std_normal)
        axs[2].set_title('Random Walk with Normal Distribution')
        axs[3].set_title('Standard Deviation of Squared Positions with Normal Distribution')

        # Exponential Distribution
        self.random_rand = False
        self.random_normal = False
        self.random_exp = True
        exp_positions = np.zeros((self.repetition, self.steps + 1))
        for i in range(self.repetition):
            positions = self.walk(self.steps)
            exp_positions[i, :] = positions
            axs[4].plot(positions)
        std_exp = np.std(exp_positions**2, axis=0)
        axs[5].plot(std_exp)
        axs[4].set_title('Random Walk with Exponential Distribution')
        axs[5].set_title('Standard Deviation of Squared Positions with Exponential Distribution')

        for ax in axs:
            ax.set_xlabel('Steps')
            ax.set_ylabel('Position')

        plt.tight_layout()
        plt.show()

    def create_grid(self):
        self.boxsize = 20
        self.boxstep = self.boxsize // 20
        self.grid = np.zeros((self.boxsize, self.boxsize))
        for i in range(self.repetition):
            pass

            

if __name__ == '__main__':
    walker = RandomWalk1D(p=0.5, repetition=10000, steps=1000)
    walker.usefunction()