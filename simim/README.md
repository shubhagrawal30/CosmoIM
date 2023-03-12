# Copied from a local copy branch 'shubh' on tim-balloon/simim on March 12, 2023
## Untracked changes made from that local copy that were not pushed to GitHub:
        modified:   SLIM-TIM/rough_cube.py
        
        modified:   TIM_model/0.1_tim_model_dev.ipynb
        
        modified:   TIM_model/1.1_making_a_tim_light_cone_and_cube.ipynb
        
        modified:   TIM_model/1.2_interacting_with_a_cube.ipynb
        
        modified:   examples/power_spectrum_demo.ipynb
        
        modified:   examples/sim_handler_demo.ipynb
        
        modified:   simim/_paths.py
        
        modified:   simim/siminterface/_rawsiminterface.py
        
        modified:   simim/siminterface/_sims.py
        
        modified:   simim/siminterface/illustris.py

-----------------------------------------

# Overview

# Installation

You may install `simim` with `pip`. If you use Python/conda environments, you may create a new one or install with `pip` inside an existing one.

`cd simim`

`pip install -e .`

## Dependencies

### Other Packages

`simim` depends on some typical scientific Python packages. You may encounter import errors if your environment does not have them installed yet.

Either of the following would work:

`pip install scipy astropy matplotlib requests h5py gdown`

`pip install -r requirements.txt`

### Data Files

Run the setup script:

`python simim/setupimsim.py`

Point to a location to house data for star formation rate lookup tables:

* Ensure `resources/sfrpaths.txt` has name, path to sfr lookup. Add line: `behroozi13 </path/to/dir/on/local/pc>/simim_resources/sfrs/behroozi13`

Point to any pre-generated light cone files:

* `touch simim/resources/lcpaths.txt`

* add line: `TNG300-1 </path/to/dir/on/local/pc>/simim_resources/lightcones/TNG300-1`

This directory should hold a `.hdf5` file with light cone data inside.

(Note: do not leave trailing new lines in these files.)

## Basic Test

`python SLIM-TIM/rough_cube.py`

should run without errors.

# Examples

See example Jupyter notebooks in `examples/`.
