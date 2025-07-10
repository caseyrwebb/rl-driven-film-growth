"""
Plotting utilities for visualizing trajectories of thickness, temperature, and
flow rate. This module provides a function to plot the trajectories of the CVD reactor's
thickness, temperature, and flow rate over time.
"""

import matplotlib.pyplot as plt


def plot_trajectories(
    thickness, temperature, flow_rate, target_thickness, title="Trajectories"
):
    """Plot thickness, temperature, and flow rate trajectories."""
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(thickness, label="Thickness (nm)")
    plt.axhline(y=target_thickness, color="r", linestyle="--", label="Target")
    plt.xlabel("Step")
    plt.ylabel("Thickness (nm)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(temperature, label="Temperature (K)")
    plt.xlabel("Step")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(flow_rate, label="Flow Rate (sccm)")
    plt.xlabel("Step")
    plt.ylabel("Flow Rate (sccm)")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
