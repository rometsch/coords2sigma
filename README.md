# coords2sigma

This is a small library providing functions to convert a 2D distribution of particles to a surface density, represented by a field on a grid.
Cartesian (x,y) and polar coordinates (r, phi) are supported.

## Functionality

- digitize 2d corrdinates and convert to surface density
- downsample an existing cartesian or polar grid (with or without logarithmic radial spacing)

## Usage

Please see the example Jupyter notebook on how to use the functions.

## Setup

Install the library by running
``` bash
python3 -m pip install git+https://github.com/rometsch/coords2sigma
```

## License

This work is published under the GNU AGPL v3 license as defined in the LICENSE document.