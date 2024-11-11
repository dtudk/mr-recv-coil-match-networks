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
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import abc
from functools import cached_property

class PreampDecoupling(ABC):
  """An abstract class for general preamplifier decoupling conditions
  """

  def __init__(self, z_coil: complex, z_out: complex, z_amp: complex):
    """Constructs a high input impedance preamplifier decoupling configuration

    Args:
        z_coil (complex): The coil impedance in Ohms
        z_out (complex): The optimal output impedance for noise-matching in Ohms
        z_amp (complex): The preamplifier input impedance in Ohms
    """
    self.__z_coil = z_coil
    self.__z_out = z_out
    self.__z_amp = z_amp
  
  @cached_property
  def z_coil(self) -> complex:
    """The coil impedance in Ohms

    Note: Should not be mutated
    """
    return self.__z_coil
  
  @cached_property
  def z_amp(self) -> complex:
    """The preamplifier input impedance in Ohms

    Note: Should not be mutated
    """
    return self.__z_amp
  
  @cached_property
  def z_out(self) -> complex:
    """The optimal output impedance for noise-matching in Ohms

    Note: Should not be mutated
    """
    return self.__z_out

  @cached_property
  def rho_out(self) -> complex:
    """The power-wave reflection coefficient at the output of a noise-matched matching network

    Note: Should not be mutated
    """
    return (self.z_out - np.conj(self.z_amp)) / (self.z_out + self.z_amp)

  @property
  @abstractmethod
  def reactance_parameters(self) -> abc.Sequence[tuple[float, float, float]]:
    """A sequence of solutions for the reactance parameters (x11, x12, x22) for an ideal matching network in Ohm

    Note: Should not be mutated
    """
    pass
  
  @property
  @abstractmethod
  def ideal_zin(self) -> complex:
    """The input impedance of an ideal matching network with this preamp decoupling configuration in Ohms

    Note: Should not be mutated
    """
    pass
  
  @property
  @abstractmethod
  def theta(self) -> float:
    """The argument of the power-wave reflection coefficient at the input port of an ideal matching network in radians

    Note: Should not be mutated
    """
    pass
  
  def calculate_preamplifier_decoupling(self, rho_in: complex) -> float:
    """Calculates the preamplifier decoupling

    Args:
        rho_in (complex): The power-wave reflection coefficient at the input of a matching network

    Returns:
        float: The preamplifier decoupling in decibels
    """
    decoupling = -20 * np.log10(np.abs(1 - rho_in * np.exp(-1j * self.theta)))
    return decoupling
  
  def calculate_theta_bounds(self, decoupling_bound: float) -> tuple[float, float]:
    """Calculates the bounds of the phase of the input power-wave reflection coefficient
    such that a given decoupling bound can still be achieved

    Args:
        decoupling_bound (float): The decoupling bound in decibels

    Returns:
        tuple[float, float]: The lower and upper bounds of the phase in radians
    """
    decoupling_bound_lin = 10 ** (decoupling_bound / -20)
    x = (np.abs(self.rho_out) ** 2 + 1 - decoupling_bound_lin ** 2) / 2
    y = np.sqrt(np.abs(self.rho_out) ** 2 - x ** 2)
    delta_theta = np.arctan2(y, x)
    lower_theta = self.theta - delta_theta
    upper_theta = self.theta + delta_theta
    return (lower_theta, upper_theta)
  
  @cached_property
  def minimum_preamplifier_decoupling(self) -> float:
    """The minimum value of preamplifier decoupling in decibels assuming a lossless matching network

    Note: Should not be mutated
    """
    decoupling = -20 * np.log10(1 + np.abs(self.rho_out))
    return decoupling
  
  @cached_property
  def maximum_preamplifier_decoupling(self) -> float:
    """The maximum value of preamplifier decoupling in decibels assuming a lossless matching network

    Note: Should not be mutated
    """
    decoupling = -20 * np.log10(1 - np.abs(self.rho_out))
    return decoupling
  
  @cached_property
  def beta(self) -> float:
    """The power standing wave ratio at the output port of a noise-matched matching network

    Note: Should not be mutated
    """
    return (1 + np.abs(self.rho_out)) / (1 - np.abs(self.rho_out))
  
  def clone(self, z_coil: complex = None, z_out: complex = None, z_amp: complex = None) -> "PreampDecoupling":
    """Clones this preamp decoupling instance with new configuration parameters

    Args:
        z_coil (complex, optional): The coil impedance in Ohms. Defaults to None.
        z_out (complex, optional): The optimal source impedance for noise-matching in Ohms. Defaults to None.
        z_amp (complex, optional): The amplifier input impedance in Ohms. Defaults to None.
    
    Note: If any arguments are not specified, the attributes for this instance are used instead.

    Note: If a subclass overrides the constructor, this method must be overridden.

    Returns:
        PreampDecoupling: A clone of this instance with new configuration parameters.
    """
    z_coil = self.z_coil if z_coil is None else z_coil
    z_out = self.z_out if z_out is None else z_out
    z_amp = self.z_amp if z_amp is None else z_amp
    return self.__class__(z_coil, z_out, z_amp)

class HighInputImpedancePreampDecoupling(PreampDecoupling):
  """Represents the high input impedance preamplifier decoupling case from [1]

  [1] W. Wang, V. Zhurbenko, J. D. Sánchez‐Heredia, and J. H. Ardenkjær‐Larsen, "Trade‐off between preamplifier noise figure and decoupling in MRI detectors," Magnetic Resonance in Medicine, vol. 89, no. 2, pp. 859–871, 2023. doi:10.1002/mrm.29489 
  """

  def __init__(self, z_coil: complex, z_out: complex, z_amp: complex):
    """Constructs a high input impedance preamplifier decoupling configuration

    Args:
        z_coil (complex): The coil impedance in Ohms
        z_out (complex): The optimal output impedance for noise-matching in Ohms
        z_amp (complex): The preamplifier input impedance
    
    Raises:
        ValueError: If the inputs violate the constraints for which a noise-matching preamplifier decoupling matching network can be constructed
    """
    if np.isclose(z_out.imag + z_amp.imag, 0) and z_amp.real > z_out.real:
      raise ValueError("The inputs violate the constraints for which a noise-matching preamplifier decoupling matching network can be constructed")
    super().__init__(z_coil, z_out, z_amp)
  
  @cached_property
  def ideal_zin(self) -> complex:
    return complex(self.beta * self.z_coil.real, -self.z_coil.imag)

  @cached_property
  def theta(self) -> float:
    return 0
  
  @cached_property
  def reactance_parameters(self) -> abc.Sequence[tuple[float, float, float]]:
    r_coil, x_coil = self.z_coil.real, self.z_coil.imag
    r_out, x_out = self.z_out.real, self.z_out.imag
    r_amp, x_amp = self.z_amp.real, self.z_amp.imag
    x11 = -x_coil + (self.beta * r_coil * (x_amp + x_out)) / (r_amp - self.beta * r_out)
    x12 = np.sqrt(r_coil * r_out * (1 + ((self.beta * (x_amp + x_out)) / (r_amp - self.beta * r_out)) ** 2))
    x22 = -x_amp + (r_amp * (x_amp + x_out)) / (r_amp - self.beta * r_out)
    return [(x11, x12, x22), (x11, -x12, x22)]

class LowInputImpedancePreampDecoupling(PreampDecoupling):
  """Represents the low input impedance preamplifier decoupling case from [1]

  [1] W. Wang, V. Zhurbenko, J. D. Sánchez‐Heredia, and J. H. Ardenkjær‐Larsen, "Trade‐off between preamplifier noise figure and decoupling in MRI detectors," Magnetic Resonance in Medicine, vol. 89, no. 2, pp. 859–871, 2023. doi:10.1002/mrm.29489 
  """
  # TODO get_reactance_parameters
  
  def __init__(self, z_coil: complex, z_out: complex, z_amp: complex):
    """Constructs a low input impedance preamplifier decoupling configuration

    Args:
        z_coil (complex): The coil impedance in Ohms
        z_out (complex): The optimal output impedance for noise-matching in Ohms
        z_amp (complex): The preamplifier input impedance in Ohms
    
    Raises:
        ValueError: If the inputs violate the constraints for which a noise-matching preamplifier decoupling matching network can be constructed
    """
    if np.isclose(z_out.imag + z_amp.imag, 0) and z_amp.real < z_out.real:
      raise ValueError("The inputs violate the constraints for which a noise-matching preamplifier decoupling matching network can be constructed")
    super().__init__(z_coil, z_out, z_amp)

  @cached_property
  def theta(self) -> float:
    return np.pi
  
  @cached_property
  def ideal_zin(self) -> complex:
    return complex(self.z_coil.real / self.beta, -self.z_coil.imag)
  
  @cached_property
  def reactance_parameters(self) -> abc.Sequence[tuple[float, float, float]]:
    r_coil, x_coil = self.z_coil.real, self.z_coil.imag
    r_out, x_out = self.z_out.real, self.z_out.imag
    r_amp, x_amp = self.z_amp.real, self.z_amp.imag
    x11 = -x_coil + (r_coil * (x_amp + x_out)) / (self.beta * r_amp - r_out)
    x12 = np.sqrt(r_coil * r_out * (1 + ((x_amp + x_out) / (self.beta * r_amp - r_out)) ** 2))
    x22 = -x_amp + (self.beta * r_amp * (x_amp + x_out)) / (self.beta * r_amp - r_out)
    return [(x11, x12, x22), (x11, -x12, x22)]