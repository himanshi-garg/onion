# ONION
A Surface Extraction Code for Planetary Discs, based on the theoretical method outlined in Pinte et al. 2018.  

Extracts altitude, velocity and temperature profiles as functions of radius for the emitting surfaces.  

Additionally, provides quick estimates for the position angle, systemic velocity, dynamical centre, and near side of the disc.

<img src="https://github.com/himanshi-garg/onion/blob/main/supplementary/shrek.jpg" width="550" height="400">

## Installation:
```bash
git clone https://github.com/himanshi-garg/onion.git
cd onion
python3 setup.py install
```

## Usage:
```bash
import onion as onion
onion.extract_layers('<fits file>', inc=<source inclination>)
```
