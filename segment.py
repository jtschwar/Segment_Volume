from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
from skimage.segmentation import flood
import ipywidgets as widgets
import numpy as np
import h5py

__all__ = [
    'panhandler',
    'image_segmenter'
]

class panhandler:
    """
    enable right click to pan image
    this doesn't set up the eventlisteners, whatever calls this needs to do
    fig.mpl_connect('button_press_event', panhandler.press)
    fig.mpl_connect('button_release_event', panhandler.release)
    
    or somehitng 
    """
    def __init__(self, figure):
        self.figure = figure
        self._id_drag = None

    def _cancel_action(self):
        self._xypress = []
        if self._id_drag:
            self.figure.canvas.mpl_disconnect(self._id_drag)
            self._id_drag = None
        
    def press(self, event):
        if event.button == 1:
            return
        elif event.button == 3:
            self._button_pressed = 1
        else:
            self._cancel_action()
            return

        x, y = event.x, event.y

        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event) and
                    a.get_navigate() and a.can_pan()):
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
                self._id_drag = self.figure.canvas.mpl_connect(
                    'motion_notify_event', self._mouse_move)
    def release(self, event):
        self._cancel_action()
        self.figure.canvas.mpl_disconnect(self._id_drag)


        for a, _ind in self._xypress:
            a.end_pan()
        if not self._xypress:
            self._cancel_action()
            return
        self._cancel_action()

    def _mouse_move(self, event):
        for a, _ind in self._xypress:
            # safer to use the recorded button at the _press than current
            # button: # multiple button can get pressed during motion...
            a.drag_pan(1, event.key, event.x, event.y)
        self.figure.canvas.draw_idle()

class image_segmenter:
    def __init__(self, in_vol, overlay_alpha=.25,figsize=(10,10),):
        """
        TODO allow for intializing with a shape instead of an image
        
        parameters
        ----------
        vol : 3D matrix
            input volume to segment
        """

        self.vol = in_vol

        self.mask_vol = np.zeros(self.vol.shape)

        (self.nx, self.ny, self.nz) = in_vol.shape

        self.shape = None        
        
        plt.ioff() # see https://github.com/matplotlib/matplotlib/issues/17013
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.gca()
        lineprops = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}
        self.lasso = LassoSelector(self.ax, self.onselect,lineprops=lineprops, button=1,useblit=False)
        self.lasso.set_visible(True)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self._release)
        self.panhandler = panhandler(self.fig)

        # setup lasso stuff
        plt.ion()

        classes = np.arange(2)
        self.colors = 'tab10'
        self.colors = np.vstack([[0,0,0],plt.get_cmap(self.colors)(np.arange(len(classes)))[:,:3]])

        print(self.colors)

        self.lasso_button = widgets.Button(
            description='lasso select',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            icon='mouse-pointer', # (FontAwesome names without the `fa-` prefix)
        )
        self.flood_button = widgets.Button(
            description='flood fill',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            icon='fill-drip', # (FontAwesome names without the `fa-` prefix)
        )
        
        self.erase_check_box = widgets.Checkbox(
            value=False,
            description='Erase Mode',
            disabled=False,
            indent=False
        )
        
        self.reset_button = widgets.Button(
            description='reset',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            icon='refresh', # (FontAwesome names without the `fa-` prefix)
        )
        self.save_button = widgets.Button(
            description='save segmented volumes',
            button_style='',
            icon='floppy-o'
        )
        self.next_button = widgets.Button(
            description='next image',
            button_style='',
            icon='arrow-right'
        )
        self.prev_button = widgets.Button(
            description='previous image',
            button_style='',
            icon='arrow-left',
            disabled=True
        )
        self.reset_button.on_click(self.reset)
        self.save_button.on_click(self.save_mask)
        self.next_button.on_click(self._change_image_idx)
        self.prev_button.on_click(self._change_image_idx)
        def button_click(button):
            if button.description == 'flood fill':
                self.flood_button.button_style='success'
                self.lasso_button.button_style=''
                self.lasso.set_active(False)
            else:
                self.flood_button.button_style=''
                self.lasso_button.button_style='success'
                self.lasso.set_active(True)
        
        self.lasso_button.on_click(button_click)
        self.flood_button.on_click(button_click)
        self.overlay_alpha = overlay_alpha
        self.indices = None
        self.new_image(0)
        

    def _change_image_idx(self, button):
        if button is self.next_button:
            if self.img_idx +1 < self.nx:
                self.img_idx += 1
                self.new_image(self.img_idx)
                
                if self.img_idx == self.nx:
                    self.next_button.disabled = True
                self.prev_button.disabled=False
        elif button is self.prev_button:
            if self.img_idx>=1:
                self.img_idx -= 1
                self.new_image(self.img_idx)
                
                if self.img_idx == 0:
                    self.prev_button.disabled=True
                
                self.next_button.disabled=False
            
    def new_image(self, img_idx):
        self.indices=None
        self.img = self.vol[img_idx,]
        # self.img = io.imread(self.image_paths[img_idx])
        self.img_idx = img_idx
        # img_path = self.image_paths[self.img_idx]
        self.ax.set_title('Slice #: {}/{}'.format(img_idx,self.nx))
        
        if self.img.shape != self.shape:
            self.shape = self.img.shape
            pix_x = np.arange(self.shape[0])
            pix_y = np.arange(self.shape[1])
            xv, yv = np.meshgrid(pix_y,pix_x)
            self.pix = np.vstack( (xv.flatten(), yv.flatten()) ).T
            self.displayed = self.ax.imshow(self.img,vmax=np.max(self.vol))
            #ensure that the _nav_stack is empty
            self.fig.canvas.toolbar._nav_stack.clear()
            #add the initial view to the stack so that the home button works.
            self.fig.canvas.toolbar.push_current()

        else:
            self.displayed.set_data(self.img)
            self.fig.canvas.toolbar.home()

        # create mask
        self.mask = self.mask_vol[img_idx,]

        self.updateArray()

    def _release(self, event):
        self.panhandler.release(event)

    def reset(self,*args):

        self.displayed.set_data(self.img)
        self.mask[:,:] = -1
        self.fig.canvas.draw()

    def onclick(self, event):
        """
        handle clicking to remove already added stuff
        """
        if event.button == 1:
            if event.xdata is not None and not self.lasso.active:
                # transpose x and y bc imshow transposes
                self.indices = flood(self.mask,(np.int(event.ydata), np.int(event.xdata)))
                self.updateArray()
        elif event.button == 3:
            self.panhandler.press(event)

    def updateArray(self):
        array = self.displayed.get_array().data

        if self.erase_check_box.value:
            if self.indices is not None:
                self.mask[self.indices] = 0
                array[self.indices] = self.img[self.indices]
        elif self.indices is not None:
            self.mask[self.indices] =  1
            # self.mask[np.where(self.img[self.mask] == 0)] = 0
        #     # https://en.wikipedia.org/wiki/Alpha_compositing#Straight_versus_premultiplied           
            c_overlay = self.mask[self.indices]*255*self.overlay_alpha
            array[self.indices] = (c_overlay + self.img[self.indices]*(1-self.overlay_alpha))
        else:
            # new image and we found a class mask
            # so redraw entire array where class != 0
            idx = self.mask != 0
            array[idx] = self.img[idx]
            c_overlay = self.mask[idx]*255*self.overlay_alpha
            array[idx] = (c_overlay + self.img[idx]*(1-self.overlay_alpha))
        self.displayed.set_data(array)
        
    def onselect(self,verts):
        self.verts = verts
        p = Path(verts)

        self.indices = p.contains_points(self.pix, radius=0).reshape(256,256)

        self.updateArray()
        self.fig.canvas.draw_idle()
        
    def render(self):
        layers = [widgets.HBox([self.lasso_button, self.flood_button])]
        layers.append(widgets.HBox([self.reset_button, self.erase_check_box]))
        layers.append(self.fig.canvas)   
        layers.append(widgets.HBox([self.save_button, self.prev_button, self.next_button]))
        return widgets.VBox(layers)
    
    def save_mask(self,save_if_no_nonzero=False):

        saveFile = h5py.File('seg_au_sto.h5','w')

        saveFile.create_dataset('original_vol',data=self.vol)
        saveFile.create_dataset('mask',data=self.mask_vol,dtype=np.uint8)
        
        seg1 = self.vol * self.mask_vol
        saveFile.create_dataset('vol1',data=seg1,dtype=np.float32)

        seg2 = self.vol - seg1
        saveFile.create_dataset('vol2',data=seg2,dtype=np.float32)        

        saveFile.close()

    def _ipython_display_(self):
        display(self.render())
