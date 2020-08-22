import numpy as np
import cv2


class WorldMap:
    def __init__(self, length=4000, height=2000):
        self.length = length
        self.height = height
        self.map = None

    # create a map of given land/water ratio
    def create(self, ratio):
        self.map = np.zeros((self.height * self.length))

        # spread random pixels w/ probability ratio in 2D space
        self.map[:int(self.height*self.length*ratio)] = 1
        np.random.shuffle(self.map)
        self.map = np.reshape(self.map, (self.height, self.length))

        eroded = cv2.erode(self.map, (3, 3))
        dilated = cv2.dilate(eroded, (3, 3))

        np.save("map_raw.npy", self.map)
        np.save("map.npy", dilated)


def load(path):
    maps = np.load(path)
    # cv2.imshow("", self.map[:1000, :2000])
    # cv2.waitKey()
    return maps


def show_map(maps):
    cv2.imshow("map", maps)
    cv2.waitKey()


def erode_dilate(maps, count=1, k=11):
    for i in range(count):
        if np.random.randint(0, 1) == 1:
            eroded = cv2.erode(maps, (k, k))
            maps = cv2.dilate(eroded, (k, k))
        else:
            dilated = cv2.dilate(maps, (k, k))
            maps = cv2.erode(dilated, (k, k))
    np.save("map_edited.npy", maps)


if __name__ == "__main__":
    wm = WorldMap()
    # wm.create(0.3333)
    # wm.load("map.npy")
    # wm.show_map(wm.load("map_edited.npy"))
    wm.erode_dilate(wm.load("map.npy"), 1, 1000)
    wm.show_map(wm.load("map_edited.npy"))
