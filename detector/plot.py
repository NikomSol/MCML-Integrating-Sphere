import numpy as np
import matplotlib.pyplot as plt


class Plot:
    pass


class PlotIntegratingSphere(Plot):

    def __init__(self, positions):
        self.positions = positions

    def plot(self, storage):
        positions = self.positions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(positions, storage[0])
        ax1.legend(title='Transmition')
        ax2.plot(positions, storage[1])
        ax2.legend(title='Reflection')

    def plot_log(self, storage):
        raise NotImplementedError
