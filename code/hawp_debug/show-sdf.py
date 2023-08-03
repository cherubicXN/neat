import matplotlib.pyplot as plt 
import os
import os.path as osp
import torch
import numpy as np

PATH = osp.join('../evals/lego/outputs/0000.pth')


outputs = torch.load(PATH)
rgb = outputs['rgb_values'].clip(min=0,max=1)
sdf = outputs['sdf']


fig, axes = plt.subplots(1,2)

axes[0].imshow(outputs['rgb_values'])
axes[1].plot(sdf[0,0],'r-')
def onclick(event):
    if event.inaxes == axes[0]:
        x = int(event.xdata)
        y = int(event.ydata)
        axes[1].clear()
        axes[1].plot(sdf[y,x],'r-')
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            (event.button, event.x, event.y, event.xdata, event.ydata))
        plt.draw()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# fig.show()
plt.show()


