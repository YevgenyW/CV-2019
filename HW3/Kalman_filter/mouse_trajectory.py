import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, './kf')
from kf import KF

np.set_printoptions(precision=2)

# time interval between measurements
dt = 0.1

def init_KF():
    q = 2
    r = 0.5
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0,0,1,0], [0,0,0,1]])
    Q = q*np.array([[dt**3/3, 0, dt**2/2, 0], [0,dt**3/3, 0, dt**2/2], [dt**2/2, 0, dt, 0], [0, dt**2/2, 0, dt] ])
    H = np.array([[1, 0, 0, 0], [0,1,0,0]])
    R = (r**2)*np.array([[1, 0], [0,1]])
    kf = KF(A=A, B=None, H=H, R=R, Q=Q)
    kf.predict()
    return kf

def get_filtered_position(filter, x, y):
    # predicted = kf.predict()
    filter.update(np.array([x, y]))
    # filter.log()
    return kf.predict()


def draw_mouse_poisition(filter):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlim(0, 1920-1)
    ax.set_ylim(0, 1080-1)

    x, y = [0], [0]
    x_filtered, y_filtered = [0], [0]
    # create empty plot
    points, = ax.plot([], [], 'o')
    filtered_points, = ax.plot([], [], '-')

    # cache the background
    background = fig.canvas.copy_from_bbox(ax.bbox)

    def on_move(event):
        time.sleep(dt)
        # append event's data to lists
        x.append(event.xdata)
        y.append(event.ydata)
        # update plot's data
        points.set_data(x,y)

        filtered = get_filtered_position(filter = filter, x = event.xdata, y = event.ydata)

        x_filtered.append(filtered[0])
        y_filtered.append(filtered[1])
        filtered_points.set_data(x_filtered, y_filtered)

        # restore background
        fig.canvas.restore_region(background)
        # redraw the points
        ax.draw_artist(points)
        # redraw the filtered points
        ax.draw_artist(filtered_points)
        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()

if __name__ == '__main__':
    # init model for Kalman filter
    kf = init_KF()
    # draw mouse positions and filtered mouse position
    draw_mouse_poisition(filter = kf)
