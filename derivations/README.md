# Derivations

## Summary
This directory contains notebooks to show how the design equations for the novel MRI receive coil matching networks were found.
Notebooks are also included to show why other topologies cannot simultaneously yield noise matching, optimal preamplifier decoupling and common-mode rejection.

## Dependencies
The dependencies for the notebooks can be installed with by running `pip install -r requirements.txt` in this directory.
The requirements at [https://lcapy.readthedocs.io/en/latest/install.html](https://lcapy.readthedocs.io/en/latest/install.html) must also be installed.

## Contents
- `valid` contains notebooks with derivations for the novel five-element matching networks.
- `invalid` contains notebooks showing that other five-element topologies cannot simultaneously yield noise matching, optimal preamplifier decoupling and common-mode rejection.

## License
This folder and its contents are part of mr-recv-coil-match-networks.
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