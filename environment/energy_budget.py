import numpy as np
import map.world_map as wm
import enviroment.energy_balance as eba
import matplotlib.pyplot as plt


class EnergyBudget:
    def __init__(self, maps, energy_balance):
        self.map = np.clip(maps, 0, 1)
        self.energy_balance = energy_balance
        self.radiation_distribution = None
        self.energy_distribution = None

    def calc_energy_distribution(self):
        energy_distribution = self.map.copy()
        for latitude in range(energy_distribution.shape[0]):
            for longitude in range(energy_distribution.shape[1]):
                if energy_distribution[latitude, longitude]:
                    energy_distribution[latitude, longitude] = self.energy_balance.get_radiation()[latitude]*(1-energy_balance.get_albedo_land())
                else:
                    energy_distribution[latitude, longitude] = (1-self.energy_balance.get_albedo_sea()[latitude])*self.energy_balance.get_radiation()[latitude]
        return energy_distribution


if __name__ == "__main__":
    maps = wm.load("/home/david/Schreibtisch/PycharmProjects/evolution/map/map_nasa.npy")
    energy_balance = eba.EnergyBalance()
    eb = EnergyBudget(maps, energy_balance)
    energy_balance.set_radiation(300, 2000)
    energy_balance.set_albedo_sea(2000)
    ed = eb.calc_energy_distribution()
    np.save("radiation_power_distribution.npy", ed)
    plt.imshow(ed)
    plt.show()
