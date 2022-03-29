# Utility and helper functions for understanding/visualizing the data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage.io as io
from glob import glob
from IPython.display import HTML


def show_video(img_dir, ext='.png', save_loc=None):
    fig = plt.figure()
    
    im_names = sorted(glob(img_dir+'*' + ext))
    imgs = [] # np array of images
    frames = [] # array of generated images
    
    for filename in im_names:
        imgs.append(io.imread(filename))

    for img in imgs:
        img_plot = plt.imshow(img)
        frames.append([img_plot])
    
    ani = animation.ArtistAnimation(fig, frames, interval=33, repeat_delay=2000)
    
    if save_loc:
        ani.save(save_loc + ".mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    html = HTML(ani.to_jshtml())
    display(html)
    plt.close()

def show_video_np(imgs, ext='.png', save_loc=None):
    fig = plt.figure()
    
    frames = [] # array of generated images
    
    for img in imgs:
        img_plot = plt.imshow(img)
        frames.append([img_plot])
    
    ani = animation.ArtistAnimation(fig, frames, interval=33, repeat_delay=2000)
    
    if save_loc:
        ani.save(save_loc + ".mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    html = HTML(ani.to_jshtml())
    display(html)
    plt.close()
