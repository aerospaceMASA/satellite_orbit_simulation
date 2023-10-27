"""
    @title  宇宙機力学2 第6回課題
    @title  衛星の軌道シミュレーション
    @date   2023/10/27
    @brief
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


class SatSimulation:
    def __init__(self, **kwargs):
        self.EARTH_RADIUS = kwargs["earth_radius"]
        self.MU = kwargs["mu"]

        self.radius = kwargs["altitude"] + self.EARTH_RADIUS
        self.altitude_dot = kwargs["altitude_dot"]
        self.theta = kwargs["theta"]
        self.velosity = kwargs["velosity"]
        self.theta_dot = self.velosity / self.radius

        self.analysis_time = kwargs["analysis_time"]

    def sim_main(self):
        # [x1_0, x2_0, x3_0, x4_0]
        X = [self.radius,
             self.altitude_dot,
             self.theta,
             self.theta_dot]

        sol = solve_ivp(self.__eom, [0, self.analysis_time],
                        X, max_step=1)

        return sol.t, sol.y

    def __eom(self, t, X):
        x_1, x_2, x_3, x_4 = X

        dx_1 = x_2
        dx_2 = x_1 * x_4**2 - self.MU / x_1**2
        dx_3 = x_4
        dx_4 = - 2 * x_2 * x_4 / x_1

        return [dx_1, dx_2, dx_3, dx_4]

    def plot_polar_graph(self, t, sol):
        self.fig = plt.figure(tight_layout=True)

        earth_radius = np.full(100, self.EARTH_RADIUS)
        earth_theta = np.linspace(0, 2 * np.pi, 100)

        ax = self.fig.add_subplot(polar=True)
        ax.plot(sol[2, :], sol[0, :], "C1", label="rocket path")
        ax.plot(earth_theta, earth_radius, "C0", label="earth shape")
        ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        ax.grid(color="black", linestyle="dotted")

        plt.show()

    def plot_raw_graph(self, t, sol):
        self.fig = plt.figure(tight_layout=True)

        ax11 = self.fig.add_subplot(2, 1, 1)
        ax11.plot(t, sol[0, :] - self.EARTH_RADIUS, "C0", label=r"$r$")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$r$ [km]")
        ax11.grid(color="black", linestyle="dotted")

        ax12 = ax11.twinx()
        ax12.set_ylabel(r"$\dot{r}$ [km/s]")
        ax12.plot(t, sol[1, :], "C1", label=r"$\dot{r}$")

        h1, l1 = ax11.get_legend_handles_labels()
        h2, l2 = ax12.get_legend_handles_labels()
        ax11.legend(h1+h2, l1+l2, bbox_to_anchor=(1.1, 1.1), loc='upper left')

        ax21 = self.fig.add_subplot(2, 1, 2)
        ax21.plot(t, np.degrees(sol[2, :]), "C0", label=r"$\theta$")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$\theta$ [deg]")
        ax21.grid(color="black", linestyle="dotted")

        ax22 = ax21.twinx()
        ax22.set_ylabel(r"$\dot{\theta}$ [deg/s]")
        ax22.plot(t, np.degrees(sol[3, :]), "C1", label=r"$\dot{\theta}$")

        h1, l1 = ax21.get_legend_handles_labels()
        h2, l2 = ax22.get_legend_handles_labels()
        ax22.legend(h1+h2, l1+l2, bbox_to_anchor=(1.1, 1.1), loc='upper left')

        plt.show()

    def plot_cartesian_graph(self, t, sol):
        self.fig = plt.figure(tight_layout=True)

        ax11 = self.fig.add_subplot(3, 1, 1)
        ax11.plot(t, sol[0, :] * np.cos(sol[2, :]), "C0")
        ax11.set_xlabel(r"$t$ [s]")
        ax11.set_ylabel(r"$x$ [km]")
        ax11.grid(color="black", linestyle="dotted")

        ax21 = self.fig.add_subplot(3, 1, 2)
        ax21.plot(t, sol[0, :] * np.sin(sol[2, :]), "C0")
        ax21.set_xlabel(r"$t$ [s]")
        ax21.set_ylabel(r"$y$ [km]")
        ax21.grid(color="black", linestyle="dotted")

        ax31 = self.fig.add_subplot(3, 1, 3)
        ax31.plot(sol[0, :] * np.cos(sol[2, :]),
                  sol[0, :] * np.sin(sol[2, :]),
                  "C0")
        ax31.set_xlabel(r"$x$ [km]")
        ax31.set_ylabel(r"$y$ [km]")
        ax31.grid(color="black", linestyle="dotted")

        plt.show()

    def save_graph(self, filename):
        self.fig.savefig(f"{filename}.png", dpi=300)


if __name__ == "__main__":
    FILE_NAME = "202310221530"

    analysis_cond = {
        "altitude": 500,
        "altitude_dot": 0,
        "theta": 0,
        "velosity": 7.9,
        "earth_radius": 6378.0,
        "mu": 3.986e5,
        "analysis_time": 20000
    }

    sat_sim = SatSimulation(**analysis_cond)

    time_array, sol_array = sat_sim.sim_main()

    sat_sim.plot_raw_graph(time_array, sol_array)
    # rocksim.save_graph(FILE_NAME + "_raw")
    sat_sim.plot_polar_graph(time_array, sol_array)
    # rocksim.save_graph(FILE_NAME + "_polar")
    sat_sim.plot_cartesian_graph(time_array, sol_array)
    # rocksim.save_graph(FILE_NAME + "_xy")
