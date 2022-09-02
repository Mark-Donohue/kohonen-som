"""
File: som.py
Author: Mark Donohue

This program plots a Kohonen Self Organizing Map over various data shapes.
"""

import numpy as np
import matplotlib.pyplot as plt

class som(object):
    """
    Class for self organizing map object
    """
    
    def __init__(self, m, n):
        """
        Initializes grid size and network weights
        """
        self.u = np.random.random((m, m, n)) / 10 + .45
        self.m = m

    def plot(self, data):
        """
        Plots training data and SOM object.
        """
        
##        # Optional ring pattern plot
##        x, y = data[:,0]-.5, data[:,1]-.5
##        r = np.sqrt(x**2 + y**2)
##        inside = r < .5
##        outside = r > .2
##        ring = np.logical_and(inside, outside)
##        x = x[ring] + .5
##        y = y[ring] + .5
##        plt.scatter(x, y, s=0.2)

        # Plot data in a square shape
        plt.scatter(data[:,0], data[:,1], s=.2)

        # Plot SOM        
        for j in range(self.m):
            for k in range(self.m):
                x, y = self.u[j, k]
                plt.plot(x, y, 'ro')

        # Draw lines
        for j in range(1, self.m):
            for k in range(0, self.m):
                x1, y1 = self.u[j, k]
                x2, y2 = self.u[j-1, k]
                plt.plot([x1, x2], [y1, y2], 'b')

        # Draw lines
        for j in range(1, self.m):
            for k in range(0, self.m):
                x1, y1 = self.u[j, k]
                x2, y2 = self.u[j, k-1]
                plt.plot([x1, x2], [y1, y2], 'b')

        # Make the plot square
        plt.gca().set_aspect('equal')
        plt.title("M=10 alpha0=0.02 d0=4 T=4000")
        plt.show()

    def learn(self, data, T, alpha0, d0):
        """
        Simulates learning process for SOM
        """        
        for t in range(T):

            if t%100 == 0:
                print(t, T)
            
            f = 1 - (t/T)
            alpha = alpha0 * f
            d = int(np.ceil(d0 * f))
            e = data[np.random.randint(data.shape[0])]
            
            # Determine Winner
            jwin, kwin = self.winner(e)

            # Find neighbors
            jbeg = max(jwin - d, 0) 
            jend = min(jwin + d, self.m)
            kbeg = max(kwin - d, 0)
            kend = min(kwin + d, self.m)

            # Loop over neighbors, updating weights
            for j in range(jbeg, jend):
               for k in range(kbeg, kend):
                    self.u[j, k] += alpha * (e - self.u[j, k])            

    def winner(self, e):
        """
        Finds winning unit whose weights are closest to input 'e'
        """
        mindist = float('inf')
        jwin, kwin = 0,0
        
        for j in range(self.m):
            for k in range(self.m):

                dist = np.sum((e - self.u[j, k])**2)
                if dist < mindist:
                    
                    # Set new minimum distance 
                    mindist = dist
                    jwin, kwin = j,k
                    
        return jwin, kwin

def main():

    # Create data
    data = np.random.random((4000,2))

    # Initial SOM    
    kohonen = som(10, 2)
    kohonen.plot(data)

    # SOM after learning    
    kohonen.learn(data, 4000, .02, 4)
    kohonen.plot(data)
                      
if __name__ == "__main__":
    main()    
