# This file is part of mr-recv-coil-match-networks.
# Copyright © 2024 Technical University of Denmark (developed by Rasmus Jepsen)
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

"""This module performs calculations to calculate parameters for MR radiofrequency receive coil matching networks with optimal noise matching and decoupling.
Calculations are from [1].
[1] W. Wang, V. Zhurbenko, J. D. Sánchez‐Heredia, and J. H. Ardenkjær‐Larsen, "Trade‐off between preamplifier noise figure and decoupling in MRI detectors," Magnetic Resonance in Medicine, vol. 89, no. 2, pp. 859–871, 2023. doi:10.1002/mrm.29489 
"""

import math
import numpy as np

def calculate_beta(rout: float, xout: float, ramp: float, xamp: float) -> float:
  """Calculates beta, the power standing wave ratio at port 2 of the matching network

  Args:
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      float: beta
  """
  zout = complex(rout, xout)
  zamp = complex(ramp, xamp)
  zampc = complex(ramp, -xamp)
  gamma_out = (zout - zampc) / (zout + zamp)
  beta = (1 + abs(gamma_out)) / (1 - abs(gamma_out))
  return beta

def calculate_reactance_matrix_low_zin(rcoil: float, xcoil: float, rout: float,
    xout: float, ramp: float, xamp: float) -> tuple[float, float, float]:
  """Calculates reactance parameters for a matching network with minimal input impedance

  Args:
      rcoil (float): The coil resistance
      xcoil (float): The coil reactance
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      tuple[float, float, float]: The reactance parameters (x11, x12, x22)
  """
  if np.isclose(xamp + xout, 0) and ramp < rout:
    raise ValueError("The inputs violate the constraints for which a noise matching preamplifier decoupling matching network can be constructed")
  beta = calculate_beta(rout, xout, ramp, xamp)
  x11 = -xcoil + (rcoil * (xamp + xout)) / (beta * ramp - rout)
  x12 = math.sqrt(rcoil * rout * (1 + ((xamp + xout) /
      (beta * ramp - rout)) ** 2))
  x22 = -xamp + (beta * ramp * (xamp + xout)) / (beta * ramp - rout)
  return (x11, x12, x22)

def calculate_reactance_matrix_high_zin(rcoil: float, xcoil: float, rout: float,
    xout: float, ramp: float, xamp: float) -> tuple[float, float, float]:
  """Calculates reactance parameters for a matching network with maximal input impedance

  Args:
      rcoil (float): The coil resistance
      xcoil (float): The coil reactance
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      tuple[float, float, float]: The reactance parameters (x11, x12, x22)
  """
  if np.isclose(xamp + xout, 0) and ramp > rout:
    raise ValueError("The inputs violate the constraints for which a noise matching preamplifier decoupling matching network can be constructed")
  beta = calculate_beta(rout, xout, ramp, xamp)
  x11 = -xcoil + (beta * rcoil * (xamp + xout)) / (ramp - beta * rout)
  x12 = math.sqrt(rcoil * rout * (1 + ((beta * (xamp + xout)) /
      (ramp - beta * rout)) ** 2))
  x22 = -xamp + (ramp * (xamp + xout)) / (ramp - beta * rout)
  return (x11, x12, x22)

def calculate_ideal_high_zin(rcoil: float, xcoil: float, rout: float,
    xout: float, ramp: float, xamp: float) -> complex:
  """The input impedance of a matching network with maximal input impedance

  Args:
      rcoil (float): The coil resistance
      xcoil (float): The coil reactance
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      complex: The input impedance
  """
  beta = calculate_beta(rout, xout, ramp, xamp)
  return complex(beta * rcoil, -xcoil)

def calculate_ideal_low_zin(rcoil: float, xcoil: float, rout: float,
    xout: float, ramp: float, xamp: float) -> complex:
  """The input impedance of a matching network with minimal input impedance

  Args:
      rcoil (float): The coil resistance
      xcoil (float): The coil reactance
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      complex: The input impedance
  """
  beta = calculate_beta(rout, xout, ramp, xamp)
  return complex(rcoil / beta, -xcoil)

def calculate_reactance_matrix(rcoil: float, xcoil: float, rout: float,
    xout: float, ramp: float, xamp: float, high_zin: bool) -> tuple[float, float, float]:
  """Calculates reactance parameters for a matching network

  Args:
      rcoil (float): The coil resistance
      xcoil (float): The coil reactance
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier
      high_zin (bool): Whether the input impedance of the matching network should be maximised

  Returns:
      tuple[float, float, float]: The reactance parameters (x11, x12, x22)
  """
  if high_zin:
    return calculate_reactance_matrix_high_zin(rcoil, xcoil, rout, xout, ramp, xamp)
  else:
    return calculate_reactance_matrix_low_zin(rcoil, xcoil, rout, xout, ramp, xamp)


def calculate_ideal_zin(rcoil: float, xcoil: float, rout: float,
    xout: float, ramp: float, xamp: float, high_zin: bool) -> complex:
  """The input impedance of a matching network

  Args:
      rcoil (float): The coil resistance
      xcoil (float): The coil reactance
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier
      high_zin (bool): Whether the input impedance of the matching network should be maximised

  Returns:
      complex: The input impedance
  """
  if high_zin:
    return calculate_ideal_high_zin(rcoil, xcoil, rout, xout, ramp, xamp)
  else:
    return calculate_ideal_low_zin(rcoil, xcoil, rout, xout, ramp, xamp)

def calculate_minimum_preamplifier_decoupling(rout: float, xout: float,
    ramp: float, xamp: float) -> float:
  """Calculates the minimum value of preamplifier decoupling for a given LNA assuming a lossless matching network

  Args:
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      float: The minimum preamplifier decoupling in decibels
  """
  zout = complex(rout, xout)
  zamp = complex(ramp, xamp)
  zampc = complex(ramp, -xamp)
  gamma_out = (zout - zampc) / (zout + zamp)
  decoupling = -20 * np.log10(1 + np.abs(gamma_out))
  return decoupling

def calculate_maximum_preamplifier_decoupling(rout: float, xout: float,
    ramp: float, xamp: float) -> float:
  """Calculates the maximum value of preamplifier decoupling for a given LNA assuming a lossless matching network

  Args:
      rout (float): The output resistance of the matching network
      xout (float): The output reactance of the matching network
      ramp (float): The input resistance of the preamplifier
      xamp (float): The input reactance of the preamplifier

  Returns:
      float: The maximum preamplifier decoupling in decibels
  """
  zout = complex(rout, xout)
  zamp = complex(ramp, xamp)
  zampc = complex(ramp, -xamp)
  gamma_out = (zout - zampc) / (zout + zamp)
  decoupling = -20 * np.log10(1 - np.abs(gamma_out))
  return decoupling