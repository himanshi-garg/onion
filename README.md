# Onion

<p align="center">
  <br/>
  <img src="https://github.com/himanshi-garg/onion/blob/main/supplementary/shrek.jpg" width="600" height="400"><br/>
  <br>
  A surface extraction code for planetary discs.<br/>
  Based on the theoretical method outlined in <b>Pinte et al. 2018</b>.
  <br/>
</p>

Extracts **altitude (z)**, **velocity (v(r,z))** and **intensity (I)** profiles as functions of radius for an emitting surface.  
Also provides quick estimates for **position angle (PA)**, **systemic velocity (v0)**, **dynamical centre (y0,x0)** and the **near facing side**.

## Installation:
```bash
git clone https://github.com/himanshi-garg/onion.git
cd onion
python3 setup.py install
```

## Usage:
```bash
import onion as onion
onion.EXTRACT('fits file', inc=inclination [degrees], distance=distance [parsecs])
e.g. onion.EXTRACT('test.fits', inc=45, distance=100)
```

distance is only required for radius (r) and altitude (z) measurements in [au], else the default ["] units are used.

to apply user specified geometric parameters:
```bash
onion.EXTRACT('fits file', inc=inclination [degrees], distance=distance [parsecs],
              cx=centre_x [pixels], cy=centre_y [pixels, PA=PA [degrees], 
              vsyst=systemic velocity [m/s])
e.g. onion.EXTRACT('test.fits', inc=45, distance=100, cx=150, cy=150, PA=45, vsyst=4000)
```

## Citation:
If you use Onion in your research, please cite the github link.

## Requirements:
<table border="0">
 <tr>
    <td>[numpy](https://pypi.org/project/numpy/)</td>
    <td>[matplotlib](https://pypi.org/project/matplotlib/)</td>
 </tr>
 <tr>
    <td>[astropy](https://pypi.org/project/astropy/)</td>
    <td>[scipy](https://pypi.org/project/scipy/)</td>
 </tr>
 <tr>
    <td>[skimage](https://pypi.org/project/scikit-image/)</td>
    <td>[cmasher](https://pypi.org/project/cmasher/)</td>
 </tr>
 <tr>
    <td>[tqdm](https://pypi.org/project/tqdm/)</td>
    <td></td>
 </tr>
</table>
