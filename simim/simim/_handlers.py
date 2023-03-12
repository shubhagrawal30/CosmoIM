import os
import warnings

import h5py
import numpy as np

from simim import _paths
from simim._mplsetup import *

class handler():

    def __init__(self,path,objectname,groupname):

        self.path = path

        self.objectname = objectname
        self.groupname = groupname

        # Initialize the various types of properties and get a bit of other
        # information
        with h5py.File(self.path,'r') as file:
            self.h = file.attrs['cosmo_h']

            self.properties_all = [key for key in file[groupname].keys() if key not in ['mass_cuts']]
            self.properties_units = {key:file[groupname][key].attrs['units'] for key in self.properties_all}
            self.properties_h_dependence = {key:file[groupname][key].attrs['h dependence'] for key in self.properties_all}

            self.properties_saved = [key for key in file[groupname].keys() if key not in ['mass_cuts']]

            self.nhalos_all = file[groupname]['mass'].shape[0]

        self.properties_loaded = {}
        self.properties_generated = []

        self.extra_props = {'h':self.h}

        # Initialize the indices of active properties
        self.inds_all = np.arange(self.nhalos_all).astype('int')
        self.inds_active = np.arange(self.nhalos_all).astype('int')

    def extract_keys(self,set='any'):
        """Get the fields attached to a file

        Parameters
        ----------
        set : {'any','loaded','saved','generated'}
            What type of keys to return, default is all

        Returns
        -------
        keys
            The fields associated with the lightcone
        """

        if set == 'any':
            return self.properties_all
        elif set == 'loaded':
            return [k for k in self.properties_loaded.keys()]
        elif set == 'saved':
            return self.properties_saved
        elif set == 'generated':
            return self.properties_generated
        else:
            raise ValueError("set not recognized")

    def has_property(self, property_name):
        """Check whether a property has been loaded into memory

        Parameters
        ----------
        property_name : str
            The name of the field to be loaded

        Returns
        -------
        exists : bool
            True if the property is present, otherwise, false
        """

        if property_name in self.properties_all:
            return True
        else:
            return False

    def has_loaded(self, property_name):
        """Check whether a property has been loaded into memory

        Parameters
        ----------
        property_name : str
            The name of the field to be loaded

        Returns
        -------
        loaded : bool
            True if the property is loaded, otherwise, false
        """

        if property_name in self.properties_loaded.keys():
            return True
        else:
            return False

    def load_property(self, *property_names):
        """Load a property from file into memory

        Parameters
        ----------
        property_names : str
            The name of the field to be loaded, can give multiple

        Returns
        -------
        none
        """

        for property_name in property_names:
            # Check property exists
            if not property_name in self.properties_saved:
                raise ValueError("{} does not have property {} saved".format(self.objectname,property_name))

            # Get the values
            with h5py.File(self.path,'r') as file:
                property_value = file[self.groupname][property_name][:]
                if len(property_value) == 0:
                    property_value = property_value.flatten()

            self.properties_loaded[property_name] = property_value

    def unload_property(self, *property_names):
        """Remove a property from memory (does not erase from file on disk)

        Parameters
        ----------
        property_names : str
            The name of the field to be loaded, can give multiple

        Returns
        -------
        none
        """

        for property_name in property_names:
            if not property_name in self.properties_loaded.keys():
                warnings.warn("Property {} is not loaded".format(property_name))

            self.properties_loaded.pop(property_name, None)

    def return_property(self, property_name, use_all_inds=False, in_h_units=False):
        """Load a property from lightcone file and return

        Parameters
        ----------
        property_name : str
            The name of the field to be loaded
        use_all_inds : bool
            If True values will be returned for all halos, otherwise only
            active halos will be returned. Default is False, but by default
            all halos are active
        in_h_units : bool (default is False)
            If True, values will be returned in units including little h.
            If False, little h dependence will be removed.

        Returns
        -------
        property_values : array
            Values of the requested property
        """

        if use_all_inds:
            inds = self.inds_all
        else:
            inds = self.inds_active

        if not property_name in self.properties_all:
            raise ValueError("{} does not have this property".format(self.objectname))

        if in_h_units:
            h_exp = 0
        else:
            h_exp = self.properties_h_dependence[property_name]

        if property_name in self.properties_loaded.keys():
            return self.properties_loaded[property_name][inds] * self.h**h_exp

        elif property_name in self.properties_saved:
            with h5py.File(self.path,'r') as file:
                value = file[self.groupname][property_name][:][inds]
                if len(value) == 0:
                    value = value.flatten()
                return value * self.h**h_exp

    def make_property(self, property, rename=None, kw_remap={}, other_kws={}, overwrite=False, use_all_inds=False):
        """Use a lineprops.prop instance to evaluate a new property

        Parameters
        ----------
        property : lineprops.prop instance
            The haloprops.property instance containing the property information
            and generating function
        rename : list, optional
            List of names specifying how to rename the property from the name
            specified in the class
        kw_remap : dict, optional
            A dictinary remaping kwargs of the property generating function to
            different properties of the lightcone. By default if the function
            has kwarg 'x' it will be evaluated on lightcone property 'x', but
            passing the dictionary {'x':'y'} will result in lightcone the
            function being evaluated on lightcone property 'y'.
        other_kws : dict, optional
            A dictionary of additional keyword arguments passed directly to
            the property.prop_function call
        overwrite : bool, optional
            Default is False. If a property name is already in use and overwrite
            is False, an error will be raised. Otherwise the property will be
            overwritten.
        use_all_inds : bool
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.

        Returns
        -------
        None
        """

        # Determine what indices to use
        if use_all_inds:
            inds = self.inds_all
        else:
            inds = self.inds_active

        # Handle renaming
        if isinstance(rename,str):
            rename = [rename]

        if rename == None:
            names = property.names
        elif len(rename) != property.n_props:
            raise ValueError("Length of rename list doesn't match number of properties")
        else:
            names = rename

        # Check if property name is already assigned
        overwrite_names = []
        for name in names:
            if name in self.properties_all:
                if overwrite:
                    warnings.warn("Property {} already exists, overwriting".format(name))
                else:
                    raise ValueError("Property {} already exists".format(name))
                overwrite_names.append(True)
            else:
                overwrite_names.append(False)

        # Evaluate property and put into memory
        all_values = property.evaluate_all(self, kw_remap=kw_remap, use_all_inds=use_all_inds, kw_arguments=other_kws)

        for i in range(property.n_props):
            name = property.names[i]
            newname = names[i]

            prop_values = np.empty(self.nhalos_all)
            prop_values[:] = np.nan
            prop_values[inds] = all_values[i]

            self.properties_loaded[newname] = prop_values
            self.properties_units[newname] = property.units[name]
            self.properties_h_dependence[newname] = property.h_dependence[name]
            if not overwrite_names[i]:
                self.properties_generated.append(newname)
                self.properties_all.append(newname)

    def write_property(self,*property_names,overwrite=False):
        """Write a property from object memory onto the saved file on
        the disk

        Parameters
        ----------
        property_names : str
            The name of the field to be written, can specify multiple
        overwrite : bool, optional
            Default is False. If a property name is already in use and overwrite
            is False, an error will be raised. Otherwise the property will be
            overwritten. Be careful if you set this to True.

        Returns
        -------
        None
        """

        for property_name in property_names:
            if not property_name in self.properties_loaded.keys():
                raise ValueError("Property {} not loaded, cannot write to disk")
            else:
                property_value = self.properties_loaded[property_name]

            if property_name in self.properties_saved:
                if overwrite:
                    warnings.warn("Property {} already exists, overwriting".format(property_name))
                else:
                    raise ValueError("Property {} already exists".format(property_name))

            with h5py.File(self.path,'a') as file:
                if not property_name in file[self.groupname].keys():
                    file[self.groupname].create_dataset(property_name,data=property_value)
                else:
                    file[self.groupname][property_name][:] = property_value

                file[self.groupname][property_name].attrs['units'] = self.properties_units[property_name]
                file[self.groupname][property_name].attrs['h dependence'] = self.properties_h_dependence[property_name]
                file[self.groupname][property_name].attrs['userdefined'] = True

                if not property_name in self.properties_saved:
                    self.properties_saved.append(property_name)

    def delete_property(self,*property_names):
        """Remove a property from the saved file on the disk

        Parameters
        ----------
        property_names : str
            The name of the field to be written, can give multiple

        Returns
        -------
        None
        """

        for property_name in property_names:
            if property_name not in self.properties_saved:
                raise ValueError("Property {} is not in the saved file".format(property_name))

            with h5py.File(self.path,'a') as file:
                if property_name not in file[self.groupname].keys():
                    raise ValueError("This property cannot be deleted")
                else:
                    del file[self.groupname][property_name]

                    self.properties_saved = [key for key in file[self.groupname].keys()]
                    properties_all_new = []
                    for i in self.properties_all:
                        if i != property_name:
                            properties_all_new.append(i)
                    self.properties_all = properties_all_new

                    self.properties_units.pop(property_name,None)
                    self.properties_h_dependence.pop(property_name,None)

    def set_property_range(self,property_name=None,pmin=-np.inf,pmax=np.inf,reset=True, in_h_units=False):
        """Set a range in a given property to be the active indices. If no
        arguments are passed, this resets the active indices to all halos

        Parameters
        ----------
        property_name : str
            The name of the field to use
        pmin : float or list
            The minimum value of the property to bracket the selected range.
            If a list is given, multiple selection regions will be used. Must
            match pmax in length
        pmax : float or list
            The maximum value of the property to bracket the selected range.
            If a list is given, multiple selection regions will be used. Must
            match pmin in length
        reset : bool, optional
            If True, the active indices will be those selected between pmin and
            pmax. If False, the active indices will be that satisfy pmin<=p<=pmax
            and which were previously in the active indices (ie this allows
            selection over multiple properties.)
        in_h_units : bool (default=False)
            If True, pmin and pmax will be taken to have units including little h,
            otherwise, they will be assumed to have units with no h dependence
            (and have the correct dependency applied before setting cuts for parameters
            where the stored catalog values are in h units).

        Returns
        -------
        None
        """

        if not property_name:
            self.inds_active = np.copy(self.inds_all)

        else:
            vals = self.return_property(property_name,use_all_inds=True,in_h_units=in_h_units)
            inds = np.nonzero((vals>=pmin) & (vals<=pmax))[0]

            if reset:
                self.inds_active = inds
            else:
                self.inds_active = np.intersect1d(inds,self.inds_active)

    def eval_stat(self, stat_function, kwargs, kw_remap={}, other_kws={},
                  use_all_inds=False,
                  give_args_in_h_units=False):

        # Check that target has required fields (ie kwargs) - as written this is incompatible with remapping keywords
        # for kwarg in kwargs:
        #     if not self.has_property(kwarg) and not kwarg in self.extra_props.keys():
        #         raise ValueError("Property {} not found".format(kwarg))

        arguments = {}
        for kwarg in other_kws.keys():
            arguments[kwarg] = other_kws[kwarg]

        for kwarg in kwargs:
            if kwarg in kw_remap.keys():
                kw_use = kw_remap[kwarg]
            else:
                kw_use = kwarg

            # Check if the kwarg is in loaded or will need to be
            # retreived
            if self.has_property(kw_use):
                arguments[kwarg] = self.return_property(kw_use,use_all_inds,in_h_units=give_args_in_h_units)
            else:
                arguments[kwarg] = self.extra_props[kw_use]

        # Evaluate function
        vals = stat_function(**arguments)

        return vals

    def plot(self, xname, yname,
             use_all_inds = False,
             save=None, axkws={}, plotkws={},in_h_units=False):
        """Make a scatter plot of two properties

        Parameters
        ----------
        xname : str
            The name of the field to use as the x-value
        yname : str
            The name of the field to use as the y-value
        use_all_inds : bool or 'compare', optional
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
            If 'compare', both sets of indices will be plotted to allow
            easy comparison.
        save : str, optional
            If specified, the plot will be saved to the given location
        axkws : dict, optional
            A dictionary of keyword args and values that will be fed to
            ax.set() when creating the plot axes
        plotkws : dict, optional
            A dictionary of keyword args and values that will be fed to
            plt.plot() when creating the plot data
        in_h_units : bool (default is False)
            If True, values will be plotted in units including little h.
            If False, little h dependence will be removed.

        Returns
        -------
        None
        """

        fig,ax = plt.subplots()
        fig.subplots_adjust(left=.15,right=.95,bottom=.15,top=.95)
        ax.set(**axkws)
        
        if 'ls' not in plotkws and 'linestyle' not in plotkws:
            plotkws['ls'] = ''
        if 'marker' not in plotkws:
            plotkws['marker'] = '.'

        if use_all_inds == 'compare':
            x = self.return_property(xname,use_all_inds=True,in_h_units=in_h_units)
            y = self.return_property(yname,use_all_inds=True,in_h_units=in_h_units)
            ax.plot(x,y,color='k',**plotkws)
            x = self.return_property(xname,use_all_inds=False,in_h_units=in_h_units)
            y = self.return_property(yname,use_all_inds=False,in_h_units=in_h_units)
            ax.plot(x,y,color='r',**plotkws)
        else:
            x = self.return_property(xname,use_all_inds=use_all_inds,in_h_units=in_h_units)
            y = self.return_property(yname,use_all_inds=use_all_inds,in_h_units=in_h_units)
            ax.plot(x,y,**plotkws)

        if save != None:
            plt.savefig(save)
        plt.show()

    def hist(self, property_name,
             use_all_inds = False,
             logtransform=False, save=None, axkws={}, plotkws={},in_h_units=False):
        """Make a histogram of a property

        Parameters
        ----------
        property_name : str
            The name of the field to use
        logtransform : bool, optional
            If set to True, will take the log of the property before
            making the histogram
        use_all_inds : bool, optional
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
        save : str, optional
            If specified, the plot will be saved to the given location
        axkws : dict, optional
            A dictionary of keyword args and values that will be fed to
            ax.set() when creating the plot axes
        plotkws : dict, optional
            A dictionary of keyword args and values that will be fed to
            plt.hist() when creating the plot data
        in_h_units : bool (default is False)
            If True, values will be plotted in units including little h.
            If False, little h dependence will be removed.


        Returns
        -------
        None
        """

        x = self.return_property(property_name,use_all_inds=use_all_inds,in_h_units=in_h_units)
        if logtransform:
            x = np.log10(x)

        fig,ax = plt.subplots()
        ax.set(**axkws)
        ax.hist(x,**plotkws)
        if save != None:
            plt.savefig(save)
        plt.show()
