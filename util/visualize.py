import visdom
import numpy as np

vis = visdom.Visdom(port=8097, env='experiment_1')


def plot_image(img):
    if img.shape[-1] == 3:
        img.transpose((2, 1, 0))
    vis.image(img)

def plot_images(imgs):
    if imgs.shape[-1] == 3:
        imgs = imgs.transpose((0, 3, 1, 2))
    vis.images(imgs)