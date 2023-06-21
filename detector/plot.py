class Plot:
    pass


class PlotIntegratingSphere(Plot):

    def __init__(self, positions):
        self.positions = positions

    def plot(self, storage):
        raise NotImplementedError
