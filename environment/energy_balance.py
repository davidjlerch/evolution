import numpy as np
import scipy.stats as st


class EnergyBalance:
    def __init__(self):
        # minimum and maximum radiation power in W/m^2
        self.radiation = None
        # albedo constant for land: 0.28, sea: [0.22(10 degree), 0.12(20 degree), 0.08(30 degree), 0.05(45 degree)],
        # forest: 0.05-0.18 (according to density)
        self.albedo_land = 0.28
        self.albedo_sea = None
        self.albedo_forest = [0.18, 0.05]
        self.chemicals_yield = []

    # real values are approximated by normal distribution
    def set_albedo_sea(self, steps):
        albedo_sea = st.norm.pdf(np.linspace(-90, 90, steps), 0.05875, 20.69131)/0.08
        self.albedo_sea = albedo_sea

    def get_albedo_sea(self):
        return self.albedo_sea

    def get_albedo_land(self):
        return self.albedo_land

    def set_radiation(self, rad_max, steps):
        radiation = np.cos(np.linspace(-90, 90, steps)*np.pi/180)*rad_max
        self.radiation = radiation

    def get_radiation(self):
        return self.radiation

    def create_chemical(self):
        self.chemicals_yield.append(np.random.randint(0, 100)/100)


if __name__ == "__main__":
    eb = EnergyBalance()

