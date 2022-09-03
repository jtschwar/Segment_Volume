from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
from skimage.segmentation import flood
import ipywidgets as widgets
import numpy as np
import h5py

__all__ = [
    'image_segmenter'
]

class image_segmenter:
    def __init__(self, in_vol, save_name='seg_volume.h5', classes=2, axis=0, overlay_alpha=.25,figsize=(10,10),):
        """    
        parameters
        ----------
        vol : 3D matrix
            input volume to segment
        save_name : String
            Output file name
        classes : Int or list
            Number of classes or a list of class names
        axis : Int (0,1,2)
            Axis to slice volume
        """

        self.vol = in_vol

        self.axis = axis

        self.save_name = save_name
        try:
            file = h5py.File(save_name,'r')
            self.mask_vol = file['mask'][:]
        except:
            self.mask_vol = np.zeros(self.vol.shape)

        if self.mask_vol.shape != self.vol.shape:
            self.mask_vol[:] = np.zeros(self.vol.shape)
            print('Previous mask has different shape from input volume!!')

        (self.nx, self.ny, self.nz) = in_vol.shape

        self.shape = None        
        
        plt.ioff() # see https://github.com/matplotlib/matplotlib/issues/17013
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.gca()
        lineprops = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}
        self.lasso = LassoSelector(self.ax, self.onselect,lineprops=lineprops, button=1,useblit=False)
        self.lasso.set_visible(True)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # setup lasso stuff
        plt.ion()

        if isinstance(classes, int):
            classes = np.arange(classes)
        if len(classes)<=10:
            self.colors = 'tab10'
        elif len(classes)<=20:
            self.colors = 'tab20'

        self.colors = np.vstack([[0,0,0],plt.get_cmap(self.colors)(np.arange(len(classes)))[:,:3]])

        self.class_dropdown = widgets.Dropdown(
                options=[(str(classes[i]), i) for i in range(len(classes))],
                value=0,
                description='Class:',
                disabled=False,
            )
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
        if self.axis == 0:      self.img = self.vol[img_idx,]
        elif self.axis == 1:    self.img = self.vol[:,img_idx,]
        else:                   self.img = self.vol[:,:,img_idx]

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
        if self.axis == 0:      self.mask = self.mask_vol[img_idx,]
        elif self.axis == 1:    self.mask = self.mask_vol[:,img_idx,]
        else:                   self.mask = self.mask_vol[:,:,img_idx]
        self.updateArray()

    def reset(self,*args):

        self.displayed.set_data(self.img)
        self.mask[:,:] = 0
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

    def updateArray(self):
        array = self.displayed.get_array().data

        if self.indices is not None:
            self.mask[self.indices] =  self.class_dropdown.value + 1
        
            # https://en.wikipedia.org/wiki/Alpha_compositing#Straight_versus_premultiplied           
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

        self.indices = p.contains_points(self.pix, radius=0).reshape(self.shape[0],self.shape[1])

        self.updateArray()
        self.fig.canvas.draw_idle()
        
    def render(self):
        layers = [widgets.HBox([self.lasso_button, self.flood_button, self.reset_button])]
        layers.append(widgets.HBox([self.class_dropdown]))
        layers.append(self.fig.canvas)   
        layers.append(widgets.HBox([self.save_button, self.prev_button, self.next_button]))
        return widgets.VBox(layers)
    
    def save_mask(self,save_if_no_nonzero=False):

        saveFile = h5py.File(self.save_name,'w')

        saveFile.create_dataset('original_vol',data=self.vol)
        saveFile.create_dataset('mask',data=self.mask_vol,dtype=np.uint8)
        
        seg1 = self.vol * self.mask_vol
        saveFile.create_dataset('vol1',data=seg1,dtype=np.float32)

        seg2 = self.vol - seg1
        saveFile.create_dataset('vol2',data=seg2,dtype=np.float32)        

        saveFile.close()

    def _ipython_display_(self):
        display(self.render())
