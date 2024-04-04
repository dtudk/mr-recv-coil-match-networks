# mr-recv-coil-match-networks

## Description
This repository implements calculations for MRI receive coil matching networks.
These matching networks provide noise matching, preamplifier decoupling and common-mode rejection.

## Installation
To install the Python package (required for using the Jupyter notebook), run `python3 -m pip install mr-recv-coil-match-networks`.
Alternatively, run `pip install .` in the root directory of this repository.

To use the Jupyter notebook, the requirements at [https://lcapy.readthedocs.io/en/latest/install.html](https://lcapy.readthedocs.io/en/latest/install.html) must also be installed.

## Usage
This repository can be used interactively through the `mr_recv_coil_match_networks.ipynb` Jupyter notebook.

This repository also implements a Python package `mr_recv_coil_match_networks`, which can be used to interact with the matching networks programmatically.

## Derivations
Notebooks relating to the derivation procedure for the baluns are located in the `derivations` directory.

## Authors and acknowledgment
This repository was created by Rasmus Jepsen for the Technical University of Denmark (DTU) to supplement the research article "Baluns for Preamplifier Decoupling of Radio-Frequency Receive Coils" by Rasmus Jepsen, Jan Henrik Ardenkjær-Larsen, and Vitaliy Zhurbenko (submitted).

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 856432).

![ERC - EU flag for projects funded under FP7 or Horizon 2020](https://erc.europa.eu/sites/default/files/LOGO_ERC-FLAG_EU%20NEGATIF.jpg)

## License
A copy of the GNU Lesser General Public License version 2.1 is available in the LICENSE file.

mr-recv-coil-match-networks
Copyright © 2024 Technical University of Denmark (developed by Rasmus Jepsen)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA