import warnings
import os

import numpy as np
from scipy.signal import fftconvolve

from simim._mplsetup import *
from matplotlib import animation

# To do
# Wrappers - function for gridding, class for power spectrum
# 1. Axis transformations and apropriate transformations of data
# 5. Average instead of sum within a cell (e.g. for gridding timestreams)
# Remake gridding based on digitize, so eventually we can support non-equal pixel sizes

def axis_edges(ax):
    delta = (ax[1]-ax[0])
    ax = ax - delta/2
    ax = np.concatenate((ax,[ax[-1]+delta]))
    return ax

def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

#############################################################################
##### SECTION 1: Under the hood functionality ###############################
#############################################################################
class _grid():
    def __init__(self,n_properties,center_point,side_length,pixel_size,axunits=None,gridunits=None):
        # Handle n_properties
        self.n_properties = n_properties
        self.n_objects = 0

        # Handle center_point inputs:
        self.center_point = np.array(center_point,ndmin=1,copy=True)
        self.n_dimensions = len(self.center_point)

        # Handle side_length inputs:
        self.side_length = np.array(side_length,ndmin=1,copy=True)
        if len(self.side_length) == 1:
            self.side_length = np.ones(self.n_dimensions)*self.side_length[0]
        if len(self.side_length) != self.n_dimensions:
            raise ValueError("side_length don't match data dimensionality")

        # Handle pixel_size inputs
        self.pixel_size = np.array(pixel_size,ndmin=1,copy=True)
        if len(self.pixel_size) != self.n_dimensions:
            if len(self.pixel_size) == 1:
                self.pixel_size = np.ones(self.n_dimensions)*self.pixel_size[0]
            else:
                raise ValueError("pixel_sizes don't match data dimensionality")

        # Handle units inputs:
        if axunits is None:
            self.axunits = np.array([None for i in range(self.n_dimensions)])
            self.fourier_axunits = np.array([None for i in range(self.n_dimensions)])
        else:
            self.axunits = np.array(axunits,ndmin=1,copy=True)
            if len(self.axunits) == 1:
                self.axunits = np.array([self.axunits[0] for i in range(self.n_dimensions)])
            if len(self.axunits) != self.n_dimensions:
                raise ValueError("axunits don't match data dimensionality")
            self.fourier_axunits = np.array([i+'^-1' for i in self.axunits],ndmin=1)

        if gridunits is None:
            self.gridunits = np.array([None for i in range(self.n_properties)])
        else:
            self.gridunits = np.array(gridunits,ndmin=1,copy=True)
            if len(self.gridunits) == 1:
                self.gridunits = np.array([self.gridunits[0] for i in range(self.n_properties)])
            if len(self.gridunits) != self.n_properties:
                raise ValueError("gridunits don't match number of properties")

        # Set up the grid - make sure the side length is compatible with
        # number of pixels
        n_pixels_decimal = self.side_length/self.pixel_size
        n_pixels_ceil = np.ceil(n_pixels_decimal)
        if np.any(n_pixels_decimal != n_pixels_ceil):
            self.side_length = self.pixel_size * n_pixels_ceil
            if (not side_length is None) and (not center_point is None):
                warnings.warn("Side lengths adjusted to accomodate integer number of pixels")
        self.n_pixels = n_pixels_ceil.astype('int')

        self.axes = []
        self.axes_centers = []
        self.fourier_axes = []
        self.fourier_axes_centers = []
        for i in range(self.n_dimensions):
            axis = np.arange(self.n_pixels[i]+1,dtype='float')
            axis *= self.pixel_size[i]
            axis -= self.side_length[i]/2
            axis += self.center_point[i]
            self.axes_centers.append(axis[:-1]+self.pixel_size[i]/2)
            self.axes.append(axis)

            self.fourier_axes.append(np.fft.fftshift(np.fft.fftfreq(self.n_pixels[i]+1,self.pixel_size[i])))
            self.fourier_axes_centers.append(np.fft.fftshift(np.fft.fftfreq(self.n_pixels[i],self.pixel_size[i])))

        self.fourier_space = np.zeros(self.n_dimensions,dtype=bool)

        self.grid_active = False
        self.is_power_spectrum = False


    def _check_property_input(self,properties):
        if properties is None:
            properties = np.arange(self.n_properties)
        
        properties = np.array(properties,ndmin=1)
        if not np.all(np.isin(properties,np.arange(self.n_properties))):
            raise ValueError("Some property indices do not correspond to property indices of the grid")
        
        return properties


    def init_grid(self):
        self.grid_active = True
        self.grid = np.zeros(np.concatenate((self.n_pixels,[self.n_properties])))
        self.n_objects = 0


    def pad(self,ax,pad,val=0):
        # Pads symmetrically - specify one value in pad for each axis in ax.
        # Negative values of pad will unpad the grid.
        ax = np.array(ax,ndmin=1).astype(int)
        if np.any(ax>=self.n_dimensions) or np.any(ax<0):
            raise ValueError('specified axes not valid')

        pad = np.array(pad,ndmin=1).astype(int)
        if len(pad) == 1 and len(ax)>1:
            pad = np.ones(len(ax))*pad[0]
        elif len(pad) != len(ax):
            raise ValueError("length of ax ({}) and pad ({}) must match".format(len(ax),len(pad)))
        
        # Check we're not removing more pixels than there are:
        for i in range(len(ax)):
            if pad[i] < 0 and self.n_pixels[ax[i]] <= -2*pad[i]:
                raise ValueError("Cannot remove more pixels than present in dimension {}".format(ax[i]))

        pad_full = []
        unpad_full = []

        new_axes = [a for a in self.axes]
        new_axes_centers = [a for a in self.axes_centers]
        new_fourier_axes = [a for a in self.fourier_axes]
        new_fourier_axes_centers = [a for a in self.fourier_axes_centers]
        new_n_pixels = np.copy(self.n_pixels)
        new_side_length = np.copy(self.side_length)
        new_pixel_size = np.copy(self.pixel_size)

        for i in range(self.n_dimensions):
            if i not in ax:
                pad_full.append((0,0))
                unpad_full.append((0,0))

            elif pad[ax[i]] == 0:
                pad_full.append((0,0))
                unpad_full.append((0,0))

            else:
                if pad[ax[i]] > 0:
                    pad_full.append((pad[ax[i]],pad[ax[i]]))
                    unpad_full.append((0,0))
                elif pad[ax[i]] < 0:
                    pad_full.append((0,0))
                    unpad_full.append((-pad[ax[i]],-pad[ax[i]]))

                new_n_pixels[i] = self.n_pixels[i] + 2*pad[ax[i]]
                if self.fourier_space[i]:
                    new_side_length[i] = self.side_length[i]
                    new_pixel_size[i] = self.side_length[i] / new_n_pixels[i]
                else:
                    new_side_length[i] = new_n_pixels[i] * self.pixel_size[i]
                    new_pixel_size[i] = self.pixel_size[i]

                new_axes[i] = np.arange(new_n_pixels[i]+1,dtype='float') * new_pixel_size[i] - new_side_length[i]/2 + self.center_point[i]
                new_axes_centers[i] = new_axes[i][:-1] + new_pixel_size[i]/2

                new_fourier_axes[i] = np.fft.fftshift(np.fft.fftfreq(new_n_pixels[i]+1,new_pixel_size[i]))
                new_fourier_axes_centers[i] = np.fft.fftshift(np.fft.fftfreq(new_n_pixels[i],new_pixel_size[i]))

        # Won't pad properties array:
        pad_full.append((0,0))
        unpad_full.append((0,0))

        # Pad / unpad the relevant axes and update the grid parameters
        self.grid = np.pad(self.grid,pad_full,constant_values=val)
        self.grid = unpad(self.grid,unpad_full)

        self.axes = new_axes
        self.axes_centers = new_axes_centers
        self.fourier_axes = new_fourier_axes
        self.fourier_axes_centers = new_fourier_axes
        self.n_pixels = new_n_pixels
        self.side_length = new_side_length
        self.pixel_size = new_pixel_size

        if self.grid.shape[:-1] != tuple(self.n_pixels):
            raise ValueError("Something went wrong - grid doesn't have expected size")


    def crop(self,ax,min=None,max=None):
    
        # Still need to sort out how to set up new axes
        ax = int(ax)
        if ax < 0 or ax > self.n_dimensions-1:
            raise ValueError("Specified axis does not exist")

        # Check if this is in Fourier space
        if self.fourier_space[ax]:
            raise ValueError("cropping not supported for Fourier space (see the pad method)")
        
        # Set min and max if not specified
        if min is None:
            min = np.min(self.axes[ax])
        if max is None:
            max = np.max(self.axes[ax])
        if max <= min:
            raise ValueError("max must be greater than min")
        
        if np.all(self.axes_centers[ax]>max) or np.all(self.axes_centers[ax]<min):
            raise ValueError("No cells within specified limits")

        # Crop the grid
        self.grid = np.take(self.grid,np.nonzero((self.axes_centers[ax]>=min) & (self.axes_centers[ax]<=max))[0],axis=ax)
            
        self.axes_centers[ax] = self.axes_centers[ax][(self.axes_centers[ax]>=min) & (self.axes_centers[ax]<=max)]
        self.axes[ax] = np.concatenate((self.axes_centers[ax] - self.pixel_size[ax]/2,[np.max(self.axes_centers[ax]) + self.pixel_size[ax]/2]))
        self.n_pixels[ax] = len(self.axes_centers[ax])
        self.side_length[ax] = np.ptp(self.axes[ax])
        self.center_point[ax] = np.min(self.axes[ax]) + self.side_length[ax]/2
        self.fourier_axes[ax] = np.fft.fftshift(np.fft.fftfreq(self.n_pixels[ax]+1,self.pixel_size[ax]))
        self.fourier_axes_centers[ax] = np.fft.fftshift(np.fft.fftfreq(self.n_pixels[ax],self.pixel_size[ax]))


    def copy_axes(self,n_properties=1):
        grid_copy = _grid(n_properties=n_properties,
                          center_point=np.copy(self.center_point),
                          side_length=np.copy(self.side_length),
                          pixel_size=np.copy(self.pixel_size),
                          axunits=np.copy(self.axunits),
                          gridunits=np.copy(self.gridunits))
        
        return grid_copy


    def copy(self,properties=None):

        properties = self._check_property_input(properties)
        n_properties = len(properties)

        grid_copy = self.copy_axes(n_properties)
        
        if self.grid_active:
            grid_copy.grid = np.copy(self.grid[...,tuple(properties)])
            grid_copy.grid_active = True
            grid_copy.n_objects = self.n_objects

        grid_copy.fourier_space = np.copy(self.fourier_space)
        grid_copy.is_power_spectrum = self.is_power_spectrum

        return grid_copy


    def save(self,path,compress=False,overwrite=False):

        if compress:
            savefunc = np.savez_compressed
        else:
            savefunc = np.savez

        if os.path.exists(path) and not overwrite:
            raise ValueError("The file you are trying to create already exists")

        save_data = {'n_properties':self.n_properties,
                     'center_point':self.center_point,
                     'side_length':self.side_length,
                     'pixel_size':self.pixel_size,
                     'axunits':self.axunits,
                     'gridunits':self.gridunits,
                     'grid_active':self.grid_active,
                     'n_objects':self.n_objects,
                     'fourier_space':self.fourier_space,
                     'is_power_spectrum':self.is_power_spectrum
                     }
        if self.grid_active:
            save_data['grid'] = self.grid

        savefunc(path,**save_data)


    def add_from_cat(self,positions,values=None,new_props=False):
        """Add values to the grid

        Parameters
        ----------
        positions : array
            The positions of each object should be an n_objects x n_dimensions
            array
        values : array (optional)
            Values to grid, should be an n_objects x n_properties array. If
            values is not specified and n_properties == 1, counts in cells
            will be added to the grid
        new_props : bool (optional)
            If True, doing this will create new entries along the poperty
            dimension for each set of values given, rather than addint them
            on top of existing values. Default is False.
        """
        if np.any(self.fourier_space):
            raise ValueError("Some axes are in fourier space, cannot add new properties in map space.")

        # If the positions array is empty don't need to do much
        if len(positions) == 0:
            if positions.ndim == 1:
                positions = positions.reshape((positions.shape[0],1))

                if new_props:
                    new_n_properties = values.shape[1]
                    new_grid = np.zeros(np.concatenate((self.n_pixels,[new_n_properties])))
                    self.grid = np.concatenate((self.grid,new_grid),axis=-1)
                    self.n_properties = self.n_properties + new_n_properties

        else:
            # Otherwise make sure the positions array is in the right shape and matches the shape of the grid
            positions = np.array(positions,ndmin=1,copy=True)
            if positions.shape[0] > 0:
                if positions.shape[1] !=self.n_dimensions:
                    raise ValueError('positions (dim={}) does not have the right number of dimensions for the grid (dim={})'.format(positions.shape[1],self.n_dimensions))

            values = np.array(values,ndmin=1,copy=True)
            if values.ndim == 1:
                values = values.reshape((values.shape[0],1))

            if values.shape[1] != self.n_properties and not new_props:
                raise ValueError("Values array does not contain the correct number of properties.")
            if values.shape[0] != positions.shape[0]:
                raise ValueError("position and values array do not have equal lenght.")

            # Put data into coordinate units
            positions -= self.center_point.reshape(1,self.n_dimensions)
            positions += self.side_length.reshape(1,self.n_dimensions)/2
            positions /= self.pixel_size.reshape(1,self.n_dimensions)
            positions = np.floor(positions).astype('int')

            # Get rid of anything that doesn't fit
            values = values[(~np.any(positions<0,axis=1)) & (~np.any(positions>=self.n_pixels,axis=1))]
            positions = positions[(~np.any(positions<0,axis=1)) & (~np.any(positions>=self.n_pixels,axis=1))]

            if not new_props:
                np.add.at(self.grid,tuple(np.split(positions,self.n_dimensions,axis=1)),np.expand_dims(values,1))
            else:
                new_n_properties = values.shape[1]
                new_grid = np.zeros(np.concatenate((self.n_pixels,[new_n_properties])))
                np.add.at(new_grid,tuple(np.split(positions,self.n_dimensions,axis=1)),np.expand_dims(values,1))
                self.grid = np.concatenate((self.grid,new_grid),axis=-1)
                self.n_properties = self.n_properties + new_n_properties

            # Count the number of objects
            self.n_objects += positions.shape[0]


    def sum_properties(self,properties=None,in_place=True):
        if properties is None:
            properties = np.arange(self.n_properties)
        
        properties = np.array(properties,ndmin=1)
        if not np.all(np.isin(properties,np.arange(self.n_properties))):
            raise ValueError("Some property indices do not correspond to property indices of the grid")
        
        sum_grid = np.expand_dims(np.sum(self.grid[...,tuple(properties)],axis=-1),-1)
        if in_place:
            self.grid = np.concatenate((self.grid,sum_grid),axis=-1)
            self.n_properties += 1

            return self
            
        else:
            new_grid = self.copy_axes(n_properties=1)
            new_grid.init_grid()
            new_grid.grid = sum_grid

            return new_grid


    def sample(self,positions,properties=None):

        # Select the properties to sample
        if properties is None:
            properties = np.arange(self.n_properties,dtype=int)
        else:
            properties = np.array(properties,ndmin=1,dtype=int)
            if np.any(properties > self.n_properties-1) or np.any(properties < 0):
                raise ValueError("Property index does not exist")
        properties = tuple(properties)

        # Make sure the positions array is in the right shape and matches the dimensions of the grid
        positions = np.array(positions,ndmin=1,copy=True)
        if positions.shape[0] > 0:
            if positions.shape[1] != self.n_dimensions:
                raise ValueError('positions (dim={}) does not have the right number of dimensions for the grid (dim={})'.format(positions.shape[1],self.n_dimensions))

        # Put positions into coordinate units
        positions -= self.center_point.reshape(1,self.n_dimensions)
        positions += self.side_length.reshape(1,self.n_dimensions)/2
        positions /= self.pixel_size.reshape(1,self.n_dimensions)
        positions = np.floor(positions).astype('int')

        # Find points outside the grid area, store them, and for now set their pixels to 0
        invalid_positions = np.nonzero(np.any(positions<0,axis=1) | np.any(positions/self.n_pixels.reshape(1,self.n_dimensions)>=1,axis=1))
        positions[invalid_positions] = 0

        positions_tuple = tuple([tuple(line) for line in positions.T])
        samples = self.grid[positions_tuple]
        samples[invalid_positions] = np.nan
        samples = samples[:,properties]

        # This is to make sure all samples have a sample number and a
        # property dimension.
        if samples.ndim == 1:
            samples = np.expand_dims(samples,-1)
        
        return samples


    def convolve(self,source,ax=None,in_place=True,pad=None):

        # Check inputs
        if not self.grid_active:
            raise ValueError("No grid is initialized.")
        if np.any(self.fourier_space):
            raise ValueError("grid has some axes in Fourier space, convolution requires all axes be in map space")
        if not source.grid_active:
            raise ValueError("No source grid is initialized.")
        if self.is_power_spectrum:
            raise ValueError("This grid is a power spectrum, not a map")

        # If the grid we convolve with has higher dimension than self, raise an error
        if source.grid.ndim > self.grid.ndim:
            raise ValueError("source grid has too many dimensions")
        # Otherwise, if the grid we convolve with has lower dimensions that self,
        # we will pad its dimensions
        elif source.grid.ndim != self.grid.ndim:
            nsourcepad = self.grid.ndim - source.grid.ndim
            shape = np.concatenate((source.grid.shape[:-1],
                                    [1 for i in range(nsourcepad)],
                                    [source.grid.shape[-1]])) # The final axis is always the 'property' axis
        else:
            nsourcepad = 0
            shape = source.grid.shape

        if ax is None:
            ax = np.arange(self.n_dimensions-nsourcepad).astype('int')
            # ax = np.arange(self.n_dimensions-1).astype('int')
        else:
            ax = np.array(ax,ndmin=1,dtype='int')
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        # Pad the array
        if pad is not None:
            if not isinstance(pad,int):
                if len(pad) != len(ax):
                    raise ValueError("pad must have one entry for each axis specified in ax")
        
        if isinstance(pad,int):
            pad = np.array([pad for i in range(len(ax))]).astype(int)
        elif pad is None:
            pad = np.array([0 for i in range(len(ax))]).astype(int)

        # Do the dang thing
        if in_place:
            self.pad(ax,pad)
            self.grid = fftconvolve(self.grid,source.grid.reshape(shape),axes=ax,mode='same')
            self.pad(ax,-pad)

            return self

        else:
            convgrid = self.copy()

            convgrid.pad(ax,pad)
            convgrid.grid = fftconvolve(convgrid.grid,source.grid.reshape(shape),axes=ax,mode='same')
            convgrid.pad(ax,-pad)

            return convgrid


    def collapse_dimension(self,ax,in_place=True,weights=None,mode='average'):
        if not self.grid_active:
            raise ValueError("No grid is initialized.")

        # Check that the grid has the specified axes
        ax = np.array(ax,ndmin=1,dtype=int)
        if np.any(ax>=self.n_dimensions) or np.any(ax<0):
            raise ValueError('specified axes not valid')
        ax_keep = np.setdiff1d(np.arange(self.n_dimensions),ax)

        # Deal with weights
        # Weights can either be None, a 1d array that will be multiplied along
        # the axis being collapsed, a self.n_dimensions array that will be multiplied
        # against each position in the grid before collapsing, or a self.n_dimensions + 1
        # array that will be multiplied across the whole grid (allowing weights to varry
        # for different properties)
        if weights is None:
            weights = np.array([np.ones(len(self.axes[i])) for i in ax])
        elif len(ax) == 1:
            weights = [weights]

        if len(weights) != len(ax):
            raise ValueError("Must provide a weights array for each axis to collapse")
        else:
            for i in range(len(ax)):
                weights[i] = np.array(weights[i])
                if weights[i].ndim == 1:
                    if len(weights[i]) != len(self.axes_centers[ax[i]]):
                        raise ValueError("Weights for axis {} must match size of grid. {} // {}".format(ax[i],len(weights[i]),len(self.axes_centers[ax[i]])))
                    else:
                        shape = [1 for i in range(self.n_dimensions)]+[1]
                        shape[ax[i]] = len(weights[i])
                        weights[i] = weights[i].reshape(tuple(shape))
                elif weights[i].ndim == self.n_dimensions:
                    if weights[i].shape != self.grid.shape[:-1]:
                        raise ValueError("Weights must match shape of grid")
                    weights[i] = np.expand_dims(weights[i],-1)
                elif weights[i].ndim == self.n_dimensions+1:
                    if weights[i].shape != self.grid.shape:
                        raise ValueError("Weights must match shape of grid")
                    weights[i] = np.expand_dims(weights[i],-1)
                else:
                    raise ValueError("Weights for axis {} must match size of grid. {} // {}".format(ax[i],len(weights[i]),len(self.axes_centers[ax[i]])))
        
        if mode=='sum':
            norm = [1 for i in ax]
        elif mode=='average':
            norm = []
            for i in range(len(ax)):
                if weights[i].ndim == 1:
                    norm.append(np.sum(weights[i]))
                else:
                    norm.append(np.sum(weights[i],axis=ax[i]))
        else:
            raise ValueError("mode not recognized - options are 'sum','average'")

        # Do the dang thing
        if in_place:
            for i in range(len(ax)):
                self.grid = np.sum(self.grid*weights[i],axis=ax[i]) / norm[i]

            # Update grid
            self.n_pixels = self.n_pixels[ax_keep]
            self.axes = [self.axes[i] for i in range(self.n_dimensions) if i not in ax]
            self.axes_centers = [self.axes_centers[i] for i in range(self.n_dimensions) if i not in ax]
            self.fourier_axes = [self.fourier_axes[i] for i in range(self.n_dimensions) if i not in ax]
            self.fourier_axes_centers = [self.fourier_axes_centers[i] for i in range(self.n_dimensions) if i not in ax]
            
            self.axunits = self.axunits[ax_keep]
            self.fourier_axunits = self.fourier_axunits[ax_keep]
            self.fourier_space = self.fourier_space[ax_keep]

            self.n_dimensions = self.n_dimensions - len(ax)
            self.center_point = self.center_point[ax_keep]
            self.side_length = self.side_length[ax_keep]
            self.pixel_size = self.pixel_size[ax_keep]

            return self

        else:
            newgrid = self.copy()

            for i in range(len(ax)):
                newgrid.grid = np.sum(newgrid.grid*weights[i],axis=ax[i]) / norm[i]

            # Update grid
            newgrid.n_pixels = self.n_pixels[ax_keep]
            newgrid.axes = [self.axes[i] for i in range(self.n_dimensions) if i not in ax]
            newgrid.axes_centers = [self.axes_centers[i] for i in range(self.n_dimensions) if i not in ax]
            newgrid.fourier_axes = [self.fourier_axes[i] for i in range(self.n_dimensions) if i not in ax]
            newgrid.fourier_axes_centers = [self.fourier_axes_centers[i] for i in range(self.n_dimensions) if i not in ax]

            newgrid.axunits = self.axunits[ax_keep]
            newgrid.fourier_axunits = self.fourier_axunits[ax_keep]
            newgrid.fourier_space = self.fourier_space[ax_keep]

            newgrid.n_dimensions = self.n_dimensions - len(ax)
            newgrid.center_point = self.center_point[ax_keep]
            newgrid.side_length = self.side_length[ax_keep]
            newgrid.pixel_size = self.pixel_size[ax_keep]

            return newgrid


    def fourier_transform(self,ax=None,normalize=True):
        """Fourier transform the grid

        This method will fourier transform the grid along specified axes.
        If the grid is already in Fourier space along a given axis, the
        inverse fourier transform will be computed instead. These changes
        are applied directly to the grid.

        Parameters
        ----------
        ax : int or list of ints (optional)
            Indices of the axes to be transformed, if no axes are given
            the operation will be applied to all axes
        normalize : bool (optional)
            Determines whether the a normalization should be applied to the
            transform (the normalization is pixel_size[i] for forward
            transforms along axis i, and 1/pixel_size[i] for inverse transforms)
        """

        if not self.grid_active:
            raise ValueError("No grid is initialized.")
        if self.is_power_spectrum:
            raise ValueError("This grid is a power spectrum, not a map")

        if ax is None:
            ax = np.arange(self.n_dimensions).astype('int')
        else:
            ax = np.array(ax,ndmin=1,dtype='int')
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        # Determine whether this is a forward or inverse FT along
        # each axis
        ax_forward = []
        norm_forward = 1
        ax_backward = []
        norm_backward = 1
        for i in ax:
            if not self.fourier_space[i]:
                ax_forward.append(i)
                norm_forward *= self.pixel_size[i]
            else:
                ax_backward.append(i)
                norm_backward /= self.pixel_size[i]

        # Do the forward FFTs
        self.grid = np.fft.fftshift(np.fft.fftn(self.grid,axes=ax_forward),axes=ax_forward)
        if normalize:
            self.grid *= norm_forward
        for i in ax_forward:
            self.fourier_space[i] = True

        # Do the inverse FFTs
        self.grid = np.fft.ifftn(np.fft.ifftshift(self.grid,axes=ax_backward),axes=ax_backward)
        if normalize:
            self.grid *= norm_backward
        for i in ax_backward:
            self.fourier_space[i] = False


    def power_spectrum(self, cross_grid=None, ax=None, in_place=False, normalize=True):
        """Compute the power spectrum of a grid in map space

        The default behavior is to return a new grid instance containing the
        power spectrum, however if in_place is set to True, the original grid
        will be replaced with the power spectrum. Note that this procedure
        cannot be reversed.

        A cross-power spectrum can be computed by specifying a cross_grid -
        another grid instance to be taken as the second term in the cross-power.

        This function checks whether a grid is already in power spectrum form,
        and will not run if it is.

        Parameters
        ----------
        cross_grid : grid instance, optional
            A second grid, which will be used to compute a cross-power spectrum
        ax : int or list of ints, optional
            Indices of the axes to be put into fourier space to compute the
            power spectrum. If no axes are given the operation will be applied
            to all axes. It does not matter what space the grid is initially
            in (coordinate of Fourier) - the method ensures that all axes
            specified in ax are transformed to Fourier space before computing
            a power spectrum. Any axes that must be transformed will be
            transformed back after the power spectrum is computed.
        in_place : bool, optional
            If set to False (default) a grid instance containing the power
            spectrum will be returned. If set to True the existing grid will
            be overwritten by the power spectrum.
        normalize : bool, optional
            Determines whether the a normalization should be applied to the
            fourier transform. See fourier_transform method for details
        """
        if not self.grid_active:
            raise ValueError("No grid is initialized.")
        if self.is_power_spectrum:
            raise ValueError("This grid is already a power spectrum")

        # Checks on the cross_grid
        if cross_grid is self:
            cross_grid = None
        if not cross_grid is None:
            if not isinstance(cross_grid,_grid):
                raise ValueError("cross_grid is not a valid grid instance")
            if not cross_grid.grid_active:
                raise ValueError("No cross_grid is initialized.")
            if cross_grid.is_power_spectrum:
                raise ValueError("This cross_grid is already a power spectrum")
            if cross_grid.n_dimensions != self.n_dimensions:
                raise ValueError("grid n_dimensions ({}) does not match cross_grid ({})".format(self.n_dimensions,cross_grid.n_dimensions))
            if cross_grid.n_properties != 1 and self.n_properties != 1:
                if cross_grid.n_properties != self.n_properties:
                    raise ValueError("grid n_properties ({}) is not compatible cross_grid ({})".format(self.n_properties,cross_grid.n_properties))

        # Determine which axes are in Fourier space already and which
        # need to be transformed
        if ax is None:
            ax = np.arange(self.n_dimensions).astype('int')
        else:
            ax = np.array(ax,ndmin=1,dtype='int')
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        transform_list = []
        for i in ax:
            if not self.fourier_space[i]:
                transform_list.append(i)
        for i in np.setdiff1d(np.arange(self.n_dimensions).astype('int'),ax):
            if self.fourier_space[i]:
                transform_list.append(i)

        # Make appropriate fourier_transform call
        self.fourier_transform(transform_list,normalize=normalize)

        # Same for cross_grid
        if not cross_grid is None:
            cross_transform_list = []
            for i in ax:
                if not cross_grid.fourier_space[i]:
                    cross_transform_list.append(i)
            for i in np.setdiff1d(np.arange(self.n_dimensions).astype('int'),ax):
                if cross_grid.fourier_space[i]:
                    cross_transform_list.append(i)

            # Make appropriate fourier_transform call
            cross_grid.fourier_transform(cross_transform_list,normalize=normalize)

        # Figure out the units to assign
        unit_end = ''
        units,n = np.unique(self.axunits[ax][self.axunits[ax] != None],return_counts=True)

        for i in range(len(units)):
            if n[i] > 1:
                unit_end += (' '+units[i]+'^'+str(n[i]))
            else:
                unit_end += (' '+units[i])

        # If in_place, then just overwrite the grid with the power spectrum
        if in_place:
            if not cross_grid is None:
                self.grid = (self.grid * cross_grid.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += cross_grid.gridunits[i] + unit_end
            else:
                self.grid = (self.grid * self.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += '^2' + unit_end

            self.is_power_spectrum = True
            # for i in range(len(self.gridunits)):
            #     if not self.gridunits[i] is None:
            #         self.gridunits += unit_end

            # Invert fourier transforms
            if not cross_grid is None:
                cross_grid.fourier_transform(cross_transform_list,normalize=normalize)

            return self

        # Else we need to create a new grid instance to hold the power
        # spectrum and return it, plus invert the fourier transforms
        # to the original grid
        else:
            powspec = _grid(self.n_properties,self.center_point,self.side_length,self.pixel_size,self.axunits,self.gridunits)

            # TO DO!! Tidy up the units
            # for i in range(len(powspec.gridunits)):
            #     if not powspec.gridunits is None:
            #         powspec.gridunits += unit_end

            if not cross_grid is None:
                powspec.grid = (self.grid * cross_grid.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += cross_grid.gridunits[i] + unit_end
            else:
                powspec.grid = (self.grid * self.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += '^2' + unit_end

            powspec.grid_active = True
            powspec.is_power_spectrum = True
            powspec.n_objects = self.n_objects
            powspec.fourier_space = np.copy(self.fourier_space)
            powspec.fourier_space[:] = True

            # Invert fourier transforms
            if not cross_grid is None:
                cross_grid.fourier_transform(cross_transform_list,normalize=normalize)
            self.fourier_transform(transform_list,normalize=normalize)

            return powspec


    def visualize(self, ax=[0,1], slice_indices='sum', property=0, figsize=(5,5), axkws={}, plotkws={}):
        fig, plot = plt.subplots(figsize=figsize)
        plot.set(xlabel='Axis {} [{}]'.format(ax[0],self.axunits[ax[0]]))
        plot.set(ylabel='Axis {} [{}]'.format(ax[1],self.axunits[ax[1]]))
        plot.set(**axkws)
        
        if len(ax) != 2:
            raise ValueError("ax must specify exactly two dimensions to plot")

        remaining_ax = [i for i in range(self.n_dimensions) if i not in ax]
        if slice_indices == 'sum':
            image = np.sum(self.grid,axis=tuple(remaining_ax))[:,:,property]
        elif len(slice_indices) != len(remaining_ax):
            raise ValueError("Slice index must be specified for all ax not plotted")
        else:
            slices = []
            slice_idx = 0
            for i in range(self.n_dimensions):
                if i in ax:
                    slices.append(slice(0,None))
                else:
                    slices.append(slice_indices[slice_idx])
                    slice_idx += 1
            image = self.grid[tuple(slices)][:,:,property]

        cb=plot.pcolor(self.axes[ax[0]],self.axes[ax[1]],image.T,**plotkws)
        fig.colorbar(cb)
        plt.show()


    def animate(self, save=None, prop_ind=0, slide_dim=2, face_dims=[0,1], i0=0, still=False, logscale=False, minrescale=1, maxrescale=1):

        if self.n_dimensions < 3:
            raise ValueError("animate only works for 3D or larger maps")

        # Check the grid is in coordinate space
        if np.any(self.fourier_space):
            warnings.warn("Some dimensions in fourier space. All values will be cast to real.")

        # handle property selection
        if prop_ind >= self.n_properties:
            raise ValueError("property index too large")

        # handle dimension selection
        if slide_dim >= self.n_dimensions:
            raise ValueError("slide_dim index too large")
        face_dims = np.array(face_dims,ndmin=1)
        if np.any(face_dims >= self.n_dimensions):
            raise ValueError("face_dims index too large")
        if len(face_dims) != 2:
            raise ValueError("Specify 2 dimensions for face_dims")

        flatten_dims = np.setdiff1d(np.arange(self.n_dimensions),np.concatenate(([slide_dim],face_dims)))

        if self.fourier_space[slide_dim]:
            ax0 = self.fourier_axes_centers[slide_dim]
        else:
            ax0 = self.axes_centers[slide_dim] - self.pixel_size[slide_dim]/2
        slide_ax = np.copy(ax0)
        ax0 = axis_edges(np.copy(ax0))

        if self.fourier_space[face_dims[0]]:
            ax1 = self.fourier_axes_centers[face_dims[0]]
        else:
            ax1 = self.axes_centers[face_dims[0]]
        ax1 = axis_edges(np.copy(ax1))

        if self.fourier_space[face_dims[1]]:
            ax2 = self.fourier_axes_centers[face_dims[1]]
        else:
            ax2 = self.axes_centers[face_dims[1]]
        ax2 = axis_edges(np.copy(ax2))

        # Determine property range allowed
        if logscale:
            vmax = np.log10(np.amax(self.grid[...,prop_ind]))
            vmin = vmax-10
        else:
            vmin = minrescale*np.amin(self.grid[...,prop_ind])
            vmax = maxrescale*np.amax(self.grid[...,prop_ind])

        # Set up plots
        figure = plt.figure(figsize=(10,5))
        title = plt.suptitle('')

        plot_edge = plt.subplot(121)
        plot_edge.set_title('Summed Along Axis {}'.format(face_dims[0]))
        # plot_edge.set(xlabel='Axis {}'.format(slide_dim),ylabel='Axis {}'.format(face_dims[1]))

        map_edge = np.sum(self.grid[...,prop_ind].real,axis=tuple(np.concatenate((flatten_dims,[face_dims[0]]))))
        if slide_dim < face_dims[1]:
            map_edge = map_edge.T
        if logscale:
            map_edge = np.log10(map_edge)
            map_edge[~np.isfinite(map_edge)] = vmin
        edge = plot_edge.pcolormesh(ax0, ax2,
                                    map_edge,
                                    cmap='inferno',
                                    vmin=np.amin(map_edge), vmax=np.amax(map_edge))
        box, = plot_edge.plot([],[],color='red',lw=.75)
        box.set_data([ax0[i0],ax0[i0],ax0[i0+1],ax0[i0+1],ax0[i0]],
                     [ax2[0],ax2[-1],ax2[-1],ax2[0],ax2[0]])

        plot_face = plt.subplot(122)
        plot_face.set_title('Axis {} = {:.2f}'.format(slide_dim,slide_ax[0]))
        # plot_face.set(xlabel='Axis {}'.format(face_dims[0]),ylabel='Axis {}'.format(face_dims[1]))

        map_face = np.take(np.sum(self.grid[...,prop_ind].real,axis=tuple(flatten_dims)),indices=i0,axis=slide_dim)
        if face_dims[0] < face_dims[1]:
            map_face = map_face.T
        if logscale:
            map_face = np.log10(map_face)
            map_face[~np.isfinite(map_face)] = vmin
        face = plot_face.pcolormesh(ax1, ax2,
                                    map_face,
                                    cmap='inferno',
                                    vmin=vmin, vmax=vmax)

        nsteps = len(slide_ax)
        def animate(i):
            plot_face.set_title('Axis {} = {:.2f}'.format(slide_dim,slide_ax[i]))
            box.set_data([ax0[i],ax0[i],ax0[i+1],ax0[i+1],ax0[i]],
                         [ax2[0],ax2[-1],ax2[-1],ax2[0],ax2[0]])
            map_face = np.take(np.sum(self.grid[...,prop_ind].real,axis=tuple(flatten_dims)),indices=i,axis=slide_dim)
            if face_dims[0] < face_dims[1]:
                map_face = map_face.T
            if logscale:
                map_face = np.log10(map_face)
                map_face[~np.isfinite(map_face)] = vmin
            face.set_array(map_face.ravel())

        if not still:
            anim = animation.FuncAnimation(figure,animate,frames=nsteps,interval=500,blit=False)
            if save != None:
                anim.save(save+'.mp4')
        else:
            if save != None:
                plt.savefig(save)

        plt.show()
        return anim


    def spherical_average(self,ax=None,center=None,bins=None,binmode='linear',weights=None,return_std=False,biased_std=True,return_n=False):
        if not self.grid_active:
            raise ValueError("No grid is initialized.")

        # If axis isn't specified use all of them
        if ax is None:
            ax = np.arange(self.n_dimensions).astype(int)
        # Otherwise check that the grid has the specified axes
        else:
            ax = np.array(ax,ndmin=1,dtype=int)
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        # Check that all axes are in either real or fourier space
        if not np.all(self.fourier_space[ax]==True) and not np.all(self.fourier_space[ax]==False):
            raise ValueError("Averaged axes must all be in either real or fourier space")
        
        # Select correct axes to define 
        if np.all(self.fourier_space[ax]==True):
            axvals = self.fourier_axes_centers
            axfourier = True
        else:
            axvals = self.axes_centers
            axfourier = False

        # If center isn't specified, use the center of the grid (spatial coordinates), or zero (fourier coordinates)
        if center is None:
            if axfourier:
                center = np.zeros(self.n_dimensions)
            else:
                center = np.copy(self.center_point)
        # Otherwise make sure it has the same dimensions as the grid
        else:
            center = np.array(center,ndmin=1)
            if len(center) != self.n_dimensions:
                raise ValueError("center dimensions don't match grid: center Nd={}, grid Nd={}".format(len(center),self.n_dimensions))
        
        # If weights isn't specified, use uniform weights
        if weights is None:
            weights = np.ones(1).reshape([1 for i in range(self.grid.ndim)])
        # Make sure weights has a shape compatible with grid
        else:
            weights = np.array(weights)
            if weights.ndim < self.grid.ndim:
                s = weights.shape
                for i in range(self.grid.ndim-weights.ndim):
                    s += (1,)
                weights = weights.reshape(s)
            for sw,sg in zip(weights.shape,self.grid.shape):
                if sw != sg and sw != 1:
                    raise ValueError("weights shape not compatible with grid: weights cast to {}, grid shape {}".format(weights.shape,self.grid.shape))

        # Compute the radius of cells using only the specified axes
        rad = np.zeros(self.grid.shape)
        for i_ax in ax:
            s = np.ones(self.n_dimensions+1,dtype=int)
            s[i_ax] = self.n_pixels[i_ax]
            rad += ((axvals[i_ax]-center[i_ax]).reshape(s))**2
        rad = np.sqrt(rad)
        rad = rad.flatten()

        # Set up bins
        if bins is None or np.issubdtype(type(bins), np.integer):

            bin_min = np.min(rad)
            bin_max = np.max(rad)
            bin_max = bin_max + (bin_max-bin_min)/1e10 # Make max slightly larger to ensure highest data point fits

            # case 1: no specifications given use min and max of axes
            if bins is None:
                bin_n = 10
            
            # Case 2: number of bins specified
            elif np.issubdtype(type(bins), np.integer):
                if bins<1:
                    raise ValueError("Positive number of bins required")
                bin_n = bins
            
            # Create bins in linear or log space
            if binmode == 'linear':
                bins = np.linspace(bin_min,bin_max,bin_n+1)
            elif binmode == 'log':
                if bin_min <=0:
                    bin_min = np.min(rad[rad>0])
                bins = np.logspace(np.log10(bin_min),np.log10(bin_max),bin_n+1)
            else:
                raise ValueError("bin mode must be in ['log','linear'].")
        
        # case 3: bins are specified by user
        else:
            bins = np.array(bins)
            if np.any(np.diff(bins)<=0):
                raise ValueError("bins must be strictly increasing")

        # Determine shape and axes of averaged data
        keep_axes = np.ones(self.n_dimensions,dtype=bool)
        keep_axes[ax] = False
        shape = [len(bins)-1] + [self.n_pixels[i_ax] for i_ax in range(self.n_dimensions) if keep_axes[i_ax]] + [self.n_properties]
        binned_shape = tuple(shape)
        binned_axes = [bins] + [axvals[i_ax] for i_ax in range(self.n_dimensions) if keep_axes[i_ax]] + [np.arange(self.n_properties)]

        # Determine the correct bin for each cell - digitize data into bins
        # and then get indices of dimensions that aren't being flattened
        bins_for_cells = np.digitize(rad,bins) - 1
        good_inds = np.nonzero((bins_for_cells>-1) & (bins_for_cells<len(bins)-2))
        coords_for_cells = np.indices(self.grid.shape)
        coords_for_cells = [bins_for_cells] + \
                           [coords_for_cells[i_ax].flatten() for i_ax in range(self.n_dimensions) if keep_axes[i_ax]] + \
                           [coords_for_cells[-1].flatten()]
        coords_for_cells = np.array(coords_for_cells).T[good_inds] # This is now an n_cells x n_dimensions (of binned data) array

        # Bin the data
        if axfourier and not self.is_power_spectrum:
            dtype_data = complex
        else:
            dtype_data = float


        # add.at needed to add items to a given index more than once in a single operation.
        # tuple(np.split(coords_for_cells,coords_for_cells.shape(1),axis=1)) creates 
        # a tuple ([cell0_x,cell1_x,...],[cell0_y,cell1_y,...],...) specifying the grid 
        # coordinates where each cell should be added
        binned_averages = np.zeros(binned_shape,dtype=dtype_data)
        np.add.at(binned_averages,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(self.grid*weights).flatten()[good_inds].reshape(-1,1))

        binned_weights = np.zeros(binned_shape,dtype=int)
        np.add.at(binned_weights,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(np.ones(self.grid.size)*weights).flatten()[good_inds].reshape(-1,1))

        binned_averages /= binned_weights

        # Count samples if requested
        if return_n:
            binned_n = np.zeros(binned_shape,dtype=int)
            np.add.at(binned_n,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(np.ones(self.grid.size)).flatten()[good_inds].reshape(-1,1))

        # Compute standard deviation if requested
        if return_std:

            good_inds_grid = np.nonzero((bins_for_cells.reshape(self.grid.shape)>-1) & (bins_for_cells.reshape(self.grid.shape)<len(bins)-2))
            means = binned_averages[tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1))].flatten()
            std_vals = np.copy(self.grid)
            std_vals[good_inds_grid] = std_vals[good_inds_grid] - means
            std_vals = std_vals**2 * weights

            binned_std = np.zeros(binned_shape,dtype=dtype_data)
            np.add.at(binned_std,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),std_vals.flatten()[good_inds].reshape(-1,1))
            
            if biased_std:
                binned_std = np.sqrt(binned_std/binned_weights)
            else:
                binned_weights2 = np.zeros(binned_shape,dtype=int)
                np.add.at(binned_weights2,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(np.ones(self.grid.size)*weights**2).flatten()[good_inds].reshape(-1,1))

                binned_std = np.sqrt(binned_std/(binned_weights-binned_weights2/binned_weights))

        if return_std and return_n:
            return binned_axes, binned_averages, binned_n, binned_std
        elif return_std:
            return binned_axes, binned_averages, binned_std
        elif return_n:
            return binned_axes, binned_averages, binned_n
        return binned_axes, binned_averages


#############################################################################
##### SECTION 2: Warppers ###################################################
#############################################################################

class gridder(_grid):
    def __init__(self,
                 positions,
                 values = None,
                 center_point = None,
                 side_length = None,
                 pixel_size = 1,
                 axunits = None,
                 gridunits = None,
                 setndims = None):
        """Put properties into a grid

        Only required argument is positions - an array of object positions.
        If no other arguments are specified, code will return a grid of
        counts in cells and will to construct a grid based on the extremal
        values of the position list and a pixel edge lenght of 1 - this will work reasonably for a densely
        populated grid, but may get poor results if the positions don't sample
        the full volume you're trying to represent.

        To specify a grid provide the center_point, side_length, and pixel_size.

        To grid a property instead of number counts, specify the values parameter
        with an array of values equal in length to positions.

        Parameters
        ----------
        positions : array
            The positions of each object should be an n_objects x n_dimensions
            array
        values : array (optional)
            Values to grid, an arbitrary number of properties can be specified
            and gridded independently. Should be an n_objects x n_properties
        center_point : array (optional)
            The center of the grid, must match the number of dimensions
            specified in positions. If left unspecified, it will be set to the
            point half way between the largest and smallest positions value
            in each dimension
        side_length : float or array (optional)
            The length of the box edges. Should either be a single value or an
            array with the same number of elements as the dimensions specified
            in positions. If left unspecified, it will be set to slightly
            larger than the stretch between the largest and smallest position
            in each dimension. If this is not an integer multiple of the
            pixel_size, it will be increased to the next integer multiple.
        pixel_size : float or array (optional)
            The length of the pixel edges. Should either be a single value or an
            array with the same number of elements as the dimensions specified
            in positions. Default is 1 in each dimension.
        axunits : str or array (optional)
            The units of each axis. Should either be a single value or an array
            with the same number of elements as the dimensions specified in
            positions
        gridunits : str or array (optional)
            The units of the grid. Should either be a single value or an array
            with the same number of elements as the number properties
        setndims : int (optional)
            The number of dimensions of the object positions - the program
            tries to determine this by default, but may be useful if the
            positions array is empty.

        Class atributes
        ---------------
        self.n_objects : int
            The number of objects with positions specified
        self.n_dimensions : int
            The number of dimensions in which the grid positions are set
        self.n_properties : int
            The number of properties specified in values (if values isn't
            specified this will be 1, for the number counts returned by
            default)
        self.pixel_size : self.n_dimensions x float array
            The pixel size in each dimension
        self.center_point : n_dimensions x float array
            The grid center point
        self.side_length : n_dimensions x float array
            The length of the box sides in each dimension
        self.n_pixels : n_dimensions x int array
            The number of pixels along each dimension
        self.axunits : n_dimensions x str array
            The units of the axes
        self.gridunits : n_properties x str array
            The units of the grid
        self.axes : list of n_dimensions arrays
            The physical values along each dimension of the grid
        self.fourier_axes : list of n_dimensions arrays
            The physical values along each dimension of the fourier transform
            of the grid
        self.fourier_space : list of n_dimensions bools
            If self.fourier_space[i] is True, the grid is in fourier space
            along the ith dimension
        self.grid : n_dimensions + 1 dimensional array
            The gridded values with the first n_dimensions axes corresponding
            to the positions, and the final axis indexing the different
            properties that have been gridded.
        """

        # Handle position inputs:
        positions = np.array(positions,ndmin=1,copy=True)
        if positions.ndim == 1:
            positions = positions.reshape((positions.shape[0],1))

        n_objects = positions.shape[0]

        if setndims is None:
            if n_objects == 0:
                raise ValueError('positions has length zero, and no dimensions are specified')
            n_dimensions = positions.shape[1]
        else:
            if not isinstance(setndims,int):
                raise ValueError('setndims must have type int')
            if n_objects == 0:
                n_dimensions = setndims
            elif setndims != positions.shape[1]:
                raise ValueError('setndims does not match the shape of the positions array')
            else:
                n_dimensions = positions.shape[1]

        # Handle value inputs:
        # If values is set to none, we'll count objects in cells
        if values is None:
            values = np.ones((n_objects,1),dtype='int')
        else:
            values = np.array(values,ndmin=1,copy=True)
            if values.ndim == 1:
                values = values.reshape((values.shape[0],1))
        n_properties = values.shape[1]

        if values.shape[0] != n_objects:
            raise ValueError("position and values array do not have equal length.")

        # Handle center_point inputs:
        if center_point is None or side_length is None:
            if n_objects == 0:
                raise ValueError("No objects provided, cannot fit a grid")
            mins = np.amin(positions,axis=0)
            ptps = np.ptp(positions,axis=0)

        if center_point is None:
            center_point = mins + ptps/2
            center_point[ptps==0] = mins[ptps==0] # In case dimension is flat
        else:
            center_point = np.array(center_point,ndmin=1,copy=True)
            if len(center_point) != n_dimensions:
                raise ValueError("center_point don't match data dimensionality")

        # Handle side_length inputs:
        if side_length is None:
            side_length = ptps * 1.0000001 # Expand slightly so both extrema get gridded
            side_length[ptps==0] = 1 # In case dimension is flat
        else:
            side_length = np.array(side_length,ndmin=1,copy=True)
            if len(side_length) == 1:
                side_length = np.ones(n_dimensions)*side_length[0]
            if len(side_length) != n_dimensions:
                raise ValueError("side_length don't match data dimensionality")

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits)
        super().init_grid()
        super().add_from_cat(positions,values)


class load_grid(_grid):
    def __init__(self,path):

        input = np.load(path,allow_pickle=True)

        super().__init__(n_properties=input['n_properties'],
                         center_point=input['center_point'],
                         side_length=input['side_length'],
                         pixel_size=input['pixel_size'],
                         axunits=input['axunits'],
                         gridunits=input['gridunits'])

        if input['grid_active']:
            self.grid_active = True
            self.grid = input['grid']
            self.n_objects = input['n_objects']
        self.fourier_space = input['fourier_space']
        self.is_power_spectrum = input['is_power_spectrum']


def gridder_function(positions, values = None, center_point = None, side_length = None, pixel_size = 1, setndims = None):
    """Wrapper for gridder to simply return the grid - see gridder docs

    Returns
    -------
    grid : n_dimensions + 1 dimensional array
        The gridded values with the first n_dimensions axes corresponding
        to the positions, and the final axis indexing the different
        properties that have been gridded.
    axes : list of n_dimensions arrays
        The physical values along each dimension of the grid
    """

    gridder_instance = gridder(positions, values, center_point, side_length, pixel_size, setndims=setndims)
    return gridder_instance.grid, gridder_instance.axes


#############################################################################
##### SECTION 3: Beams, Masks, PSFs, etc ####################################
#############################################################################

class psf(_grid):
    def __init__(self,fwhm,pixel_size,side_length=None,axunits=None,norm='area'):

        n_properties = 1

        # Check inputs
        if norm not in ['area','peak']:
            raise ValueError("norm option not recognized")

        self.fwhm = np.array(fwhm,ndmin=1,copy=True)

        n_dimensions = self.fwhm.shape[0]
        center_point = np.zeros(n_dimensions)

        if not side_length is None:
            side_length = np.array(side_length,ndmin=1)
            if len(side_length) == 1:
                side_length = np.ones(n_dimensions) * side_length
            elif len(side_length) != n_dimensions:
                raise ValueError("side_lengths does not have same dimensionality as fwhm")
        else:
            side_length = 6*self.fwhm

        pixel_size = np.array(pixel_size,ndmin=1,copy=True)
        if len(pixel_size) == 1:
            pixel_size = np.ones(n_dimensions) * pixel_size
        elif len(pixel_size) != n_dimensions:
            raise ValueError("pixel_size does not have same dimensionality as fwhm")

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits=None)

        # Make the grid
        sig = self.fwhm / (2*np.sqrt(2*np.log(2)))
        psfnd = np.ones(self.n_pixels)
        for i in range(len(self.axes)):
            psf1d = np.exp(-self.axes_centers[i]**2 / (2*sig[i]**2))
            shape = np.ones(self.n_dimensions,dtype='int')
            shape[i] = len(psf1d)
            psfnd *= psf1d.reshape(shape)
        shape = np.concatenate((psfnd.shape,[1]))
        self.grid = psfnd.reshape(shape)

        if norm == 'area':
            self.grid /= np.sum(self.grid)

        self.grid_active = True
        self.n_objects = 0


class spectralpsf(_grid):
    def __init__(self,spec_axis,spec0,fwhm0,pixel_size,side_length=None,axunits=None,norm='area'):

        n_properties = 1

        # Check inputs
        if norm not in ['area','peak']:
            raise ValueError("norm option not recognized")

        self.fwhm0 = np.array(fwhm0,ndmin=1,copy=True)
        spec_axis = np.array(spec_axis,ndmin=1,copy=True)

        n_dimensions = self.fwhm0.shape[0] + 1
        center_point = np.zeros(self.fwhm0.shape[0])
        center_point = np.concatenate((center_point,[np.ptp(spec_axis)/2+np.amin(spec_axis)]))

        if not side_length is None:
            side_length = np.array(side_length,ndmin=1)
            if len(side_length) == 1:
                side_length = np.ones(n_dimensions-1) * side_length
            elif len(side_length) != n_dimensions-1:
                raise ValueError("side_lengths does not have same dimensionality as fwhm0")
        else:
            side_length = 3*self.fwhm0 * np.amax(spec0/spec_axis)
        side_length = np.concatenate((side_length,[np.ptp(spec_axis)+np.abs(spec_axis[1]-spec_axis[0])]))

        pixel_size = np.array(pixel_size,ndmin=1,copy=True)
        if len(pixel_size) == 1:
            pixel_size = np.ones(n_dimensions-1) * pixel_size
        elif len(pixel_size) != n_dimensions-1:
            raise ValueError("pixel_size does not have same dimensionality as fwhm0")
        pixel_size = np.concatenate((pixel_size,[np.abs(spec_axis[1]-spec_axis[0])]))

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits=None)
        self.axes[-1] = spec_axis

        # Make the grid
        sig = self.fwhm0 / (2*np.sqrt(2*np.log(2)))
        psfnd = np.ones(self.n_pixels)
        for i in range(len(self.axes)-1):
            xx,ss = np.meshgrid(self.axes_centers[i],spec_axis,indexing='ij')
            psf2d = np.exp(-xx**2 / (2*(sig[i]*spec0/ss)**2))
            shape = np.ones(self.n_dimensions,dtype='int')
            shape[i] = len(self.axes_centers[i])
            shape[-1] = len(spec_axis)
            psfnd *= psf2d.reshape(shape)
        shape = np.concatenate((psfnd.shape,[1]))
        self.grid = psfnd.reshape(shape)

        if norm == 'area':
            axes = tuple(np.arange(self.n_dimensions-1,dtype='int'))
            sums = np.sum(self.grid,axis=axes)
            shape = np.concatenate((np.ones(self.n_dimensions-1,dtype='int'),[len(sums),1]))
            self.grid /= sums.reshape(shape)

        self.grid_active = True
        self.n_objects = 0



#############################################################################
##### SECTION 3: Work in progress ###########################################
#############################################################################

# Add some checks to ensure the desired axes are reproduced correctly
# Can be used as the basis for a much more generic set of PSFs
class grid_from_axes(_grid):
    def __init__(self,*axes,n_properties=1,axunits=None,gridunits=None):

        center_point = []
        side_length = []
        pixel_size = []
        for i,ax in enumerate(axes):
            steps = ax[1:]-ax[:-1]
            if not np.all(np.isclose(steps, steps[0])):
                raise ValueError("grid spacing is not uniform")
            
            center_point.append((ax[-1]+ax[0])/2)
            side_length.append(ax[-1]-ax[0])
            pixel_size.append((ax[-1]-ax[0])/(len(ax)-1))
        
        center_point = np.array(center_point)
        side_length = np.array(side_length)
        pixel_size = np.array(pixel_size)

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits)

        n_pixels_decimal = self.side_length/self.pixel_size
        n_pixels_ceil = np.ceil(n_pixels_decimal)
        if np.any(n_pixels_decimal != n_pixels_ceil):
            self.side_length = self.pixel_size * n_pixels_ceil
            if (not side_length is None) and (not center_point is None):
                warnings.warn("Side lengths adjusted to accomodate integer number of pixels")
        self.n_pixels = n_pixels_ceil.astype('int')

class grid_from_axes_and_function(grid_from_axes):
    def __init__(self,function,*axes,function_kwargs={},n_properties=1,axunits=None,gridunits=None):

        super().__init__(*axes,n_properties=n_properties,axunits=axunits,gridunits=gridunits)
        self.init_grid()
        shape = self.grid.shape
        self.grid = np.expand_dims(function(*self.axes_centers,**function_kwargs),axis=-1)
        if self.grid.shape != shape:
            raise ValueError("function does not produce a grid of the correct shape")

