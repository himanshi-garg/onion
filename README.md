# Onion

<p align="center">
<img src="https://github.com/himanshi-garg/onion/blob/main/supplementary/shrek.jpg" width="600" height="400">
</p>
<p align="center">
A surface extraction code for planetary discs.  
Based on the theoretical method outlined in **Pinte et al. 2018**.
</p>

Extracts **altitude (z)**, **velocity** and **intensity profiles** as functions of radius for an emitting surface.  
Additionally provides quick estimates for **position angle (PA)**, **systemic velocity (v0)**, **dynamical centre (y0,x0)** and the **near facing side**.

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
