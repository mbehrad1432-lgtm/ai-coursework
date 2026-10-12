"""Replacement for missing draw.py, added in the portfolio edition.

Coordinates: x increases right, y increases with row index. Beacon locations
are cell centers; beacon cells are traversable. No original draw.py was available
for comparing conventions. Uses Python's seeded random generator.
"""
import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Maze:
    def __init__(self, layout):
        if not layout or any(len(row) != len(layout[0]) for row in layout):
            raise ValueError('Maze must be a nonempty rectangle')
        self.layout = layout
        self.height, self.width = len(layout), len(layout[0])
        self.free_cells = [(c, r) for r,row in enumerate(layout) for c,value in enumerate(row) if value != 1]
        self.beacons = [(c+.5,r+.5) for r,row in enumerate(layout) for c,value in enumerate(row) if value == 2]
        if not self.free_cells or not self.beacons:
            raise ValueError('Need free cells and at least one beacon')

    def is_free(self, x, y):
        return (math.isfinite(x) and math.isfinite(y) and 0 <= x < self.width and 0 <= y < self.height and self.layout[int(y)][int(x)] != 1)

    def random_free_place(self):
        c,r = random.choice(self.free_cells)
        return c+random.random(), r+random.random()

    def distance_to_nearest_beacon(self, x, y):
        return min(math.hypot(x-bx,y-by) for bx,by in self.beacons)

    def save_snapshot(self, particles, robot, mean, path):
        fig,ax = plt.subplots(figsize=(7,7))
        walls = np.array(self.layout) == 1
        ax.imshow(walls, cmap='Greys', vmin=0,vmax=1,origin='upper',extent=(0,self.width,self.height,0))
        ax.scatter([p.x for p in particles],[p.y for p in particles],c=[p.w for p in particles],cmap='Reds',s=5,alpha=.5,label='Particles')
        ax.scatter(*zip(*self.beacons),c='cyan',edgecolors='black',s=65,label='Beacons')
        ax.scatter([robot.x],[robot.y],c='green',marker='*',s=180,label='True robot')
        ax.scatter([mean[0]],[mean[1]],c='orange',marker='X',s=110,label='Weighted mean')
        ax.set_title('Particle filter final snapshot — seed 42')
        ax.set_xlabel('x (cells)');ax.set_ylabel('y (cells)')
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1))
        fig.tight_layout();fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig)
