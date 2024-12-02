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

"""This module encapsulates the functionality of matching network topologies.
"""

import lc_power_match_baluns.oneport
import lc_power_match_baluns.topology
from abc import abstractmethod, ABC
from collections import abc
import numpy as np
import skrf
from multimethod import multimethod

class ReceiveCoilMatchingTopology(lc_power_match_baluns.topology.BalunTopology, ABC):
  """A matching network topology for an MR receive coil and a preamplifier
  
  Attributes:
        has_common_mode_rejection Whether the topology is explicitly designed to provide common-mode rejection.
        has_only_one_solution Whether two solutions of reactance parameters yield equivalent networks for this topology.
  """

  has_common_mode_rejection: bool = False

  has_only_one_solution: bool = False

  @classmethod
  @abstractmethod
  def calculate_elements_from_reactance_params(cls, x11: float, x12: float,
      x22: float) -> tuple[float, ...]:
    """Calculates element reactances from reactance parameters for optimal matching/decoupling/common-mode rejection

    Args:
        x11 (float): X11 reactance parameter in Ohms
        x12 (float): X12 reactance parameter in Ohms
        x22 (float): X22 reactance parameter in Ohms

    Raises:
        NotImplementedError: If the topology has not implemented this method

    Returns:
        tuple[float, ...]: The element reactances in Ohms
    """
    raise NotImplementedError
  
  @multimethod
  def calculate_input_impedance(cls, z_amp: complex, # pylint: disable=no-self-argument
      element_impedances: abc.Sequence[complex]) -> complex:
    """Calculates an estimate of the input impedance of the matching network

    Args:
        z_amp (complex): The amplifier input impedance in Ohms
        element_impedances (abc.Sequence[complex]): The impedances of each element in Ohms

    Returns:
        complex: The input impedance in Ohms
    """
    input_impedance = cls.calculate_input_impedance([z_amp], list(zip(element_impedances)))
    return input_impedance[0]

  @classmethod
  @multimethod
  def calculate_input_impedance(cls, z_amp: abc.Sequence[complex], # pylint: disable=function-redefined
      element_impedances: abc.Sequence[abc.Sequence[complex]]) -> abc.Sequence[complex]:
    """Calculates an estimate of the input impedance of the matching network over frequency

    Args:
        z_amp (abc.Sequence[complex]): The amplifier input impedance in Ohms over frequency
        element_impedances (abc.Sequence[abc.Sequence[complex]]): A sequence of all elements,
            with the inner sequence representing the impedance of an element in Ohms over frequency

    Returns:
        abc.Sequence[complex]: The input impedance in Ohms over frequency
    """
    amp_z = np.array(z_amp).reshape((len(z_amp), 1, 1))
    amp_network = skrf.Network.from_z(amp_z)
    twoport_z = cls.calculate_two_port_impedance_parameters(element_impedances) # pylint: disable=no-value-for-parameter
    matching_network = skrf.Network.from_z(twoport_z)
    loaded_matching_network = skrf.network.connect(amp_network, 0, matching_network, 1)
    return loaded_matching_network.z[:, 0, 0]
  
  @multimethod
  def calculate_rho_in(cls, z_coil: complex, z_amp: complex, # pylint: disable=no-self-argument
      element_impedances: abc.Sequence[complex]) -> complex:
    """Calculates an estimate of the power-wave reflection coefficient at the input port of the matching network

    Args:
        z_coil (complex): The coil impedance in Ohms
        z_amp (complex): The amplifier input impedance in Ohms
        element_impedances (abc.Sequence[complex]): The impedances of each element in Ohms

    Returns:
        complex: The power-wave reflection coefficient
    """
    rho_in = cls.calculate_rho_in([z_coil], [z_amp], list(zip(element_impedances)))
    return rho_in[0]

  @classmethod
  @multimethod
  def calculate_rho_in(cls, z_coil: abc.Sequence[complex], z_amp: abc.Sequence[complex], # pylint: disable=function-redefined
      element_impedances: abc.Sequence[abc.Sequence[complex]]) -> abc.Sequence[complex]:
    """Calculates an estimate of the power-wave reflection coefficient at the input port of the matching network over frequency

    Args:
        z_coil (abc.Sequence[complex]): The coil impedance in Ohms over frequency
        z_amp (abc.Sequence[complex]): The amplifier input impedance in Ohms over frequency
        element_impedances (abc.Sequence[abc.Sequence[complex]]): A sequence of all elements,
            with the inner sequence representing the impedance of an element in Ohms over frequency

    Returns:
        abc.Sequence[complex]: The power-wave reflection coefficient over frequency
    """
    twoport_s = cls.calculate_two_port_scattering_parameters(z_coil, z_amp, element_impedances, "power") # pylint: disable=no-value-for-parameter
    return twoport_s[:, 0, 0]

  @multimethod
  def calculate_output_impedance(cls, z_coil: complex, # pylint: disable=no-self-argument
      element_impedances: abc.Sequence[complex]) -> complex:
    """Calculates an estimate of the output impedance of the matching network

    Args:
        z_coil (complex): The coil impedance in Ohms
        element_impedances (abc.Sequence[complex]): The impedances of each element in Ohms

    Returns:
        complex: The output impedance in Ohms
    """
    output_impedance = cls.calculate_output_impedance([z_coil], list(zip(element_impedances)))
    return output_impedance[0]

  @classmethod
  @multimethod
  def calculate_output_impedance(cls, z_coil: abc.Sequence[complex], # pylint: disable=function-redefined
      element_impedances: abc.Sequence[abc.Sequence[complex]]) -> abc.Sequence[complex]:
    """Calculates an estimate of the output impedance of the matching network over frequency

    Args:
        z_coil (abc.Sequence[complex]): The coil impedance in Ohms over frequency
        element_impedances (abc.Sequence[abc.Sequence[complex]]): A sequence of all elements,
            with the inner sequence representing the impedance of an element in Ohms over frequency

    Returns:
        abc.Sequence[complex]: The output impedance in Ohms over frequency
    """
    coil_z = np.array(z_coil).reshape((len(z_coil), 1, 1))
    coil_network = skrf.Network.from_z(coil_z)
    twoport_z = cls.calculate_two_port_impedance_parameters(element_impedances) # pylint: disable=no-value-for-parameter
    matching_network = skrf.Network.from_z(twoport_z)
    loaded_matching_network = skrf.network.connect(coil_network, 0, matching_network, 0)
    return loaded_matching_network.z[:, 0, 0]
  
  @multimethod
  def calculate_available_power_gain(cls, z_coil: complex, # pylint: disable=no-self-argument
      element_impedances: abc.Sequence[complex]) -> float:
    """Calculates an estimate of the available power gain of the matching network

    Args:
        z_coil (complex): The coil impedance in Ohms
        element_impedances (abc.Sequence[complex]): The impedances of each element in Ohms
    
    Note: For a passive matching network, this should return a negative gain (in dB).
        The main purpose of this method is to provide noise figure estimates for lossy matching networks.

    Returns:
        float: The available power gain in decibels
    """
    available_power_gain = cls.calculate_available_power_gain([z_coil], list(zip(element_impedances)))
    return available_power_gain[0]

  @classmethod
  @multimethod
  def calculate_available_power_gain(cls, z_coil: abc.Sequence[complex], # pylint: disable=function-redefined
      element_impedances: abc.Sequence[abc.Sequence[complex]]) -> abc.Sequence[float]:
    """Calculates an estimate of the available power gain of the matching network over frequency

    Args:
        z_coil (abc.Sequence[complex]): The coil impedance in Ohms over frequency
        element_impedances (abc.Sequence[abc.Sequence[complex]]): A sequence of all elements,
            with the inner sequence representing the impedance of an element in Ohms over frequency
    
    Note: The frequencies for z_coil and element_impedances are assumed to be the same

    Note: For a passive matching network, this should return a negative gain (in dB).
        The main purpose of this method is to provide noise figure estimates for lossy matching networks.

    Returns:
        abc.Sequence[float]: The available power gain in decibels over frequency
    """
    y_coil = 1 / np.array(z_coil)
    y_out = 1 / np.array(cls.calculate_output_impedance(z_coil, element_impedances)) # pylint: disable=no-value-for-parameter
    twoport_z = cls.calculate_two_port_impedance_parameters(element_impedances) # pylint: disable=no-value-for-parameter
    matching_network = skrf.Network.from_z(twoport_z)
    matching_network_y = matching_network.y
    matching_network_gav = np.absolute(matching_network_y[:, 1, 0] / (matching_network_y[:, 0, 0] + y_coil)) ** 2 * np.real(y_coil) / np.real(y_out)
    return 10 * np.log10(matching_network_gav)
  
  @multimethod
  def calculate_noise_figure(cls, z_coil: complex, z_out: complex, fmin: float, rn: float, # pylint: disable=no-self-argument
      element_impedances: abc.Sequence[complex]) -> float:
    """Calculates a noise figure estimate from element impedances

    Args:
        z_coil (complex): The coil impedance in Ohms
        z_out (complex): The optimal output impedance of the matching network in Ohms
        fmin (float): The minimum noise figure for the preamplifier in decibels
        rn (float): The preamplifier noise resistance in Ohms
        element_impedances (abc.Sequence[complex]): The impedances of each element in Ohms
      
    Returns:
        float: The noise figure in decibels
    """
    nf = cls.calculate_noise_figure([z_coil], [z_out], [fmin], [rn], list(zip(element_impedances)))
    return nf[0]
  
  @classmethod
  @multimethod
  def calculate_noise_figure(cls, z_coil: abc.Sequence[complex], # pylint: disable=function-redefined
      z_out: abc.Sequence[complex], fmin: abc.Sequence[float], rn: abc.Sequence[float],
      element_impedances: abc.Sequence[abc.Sequence[complex]]) -> abc.Sequence[float]:
    """Calculates a noise figure estimate from element impedances

    Args:
        z_coil (abc.Sequence[complex]): The coil impedance in Ohms over frequency
        z_out (abc.Sequence[complex]): The optimal output impedance of the matching network in Ohms over frequency
        fmin (fmin: abc.Sequence[float]): The minimum noise figure for the preamplifier in decibels over frequency
        rn (abc.Sequence[float]): The preamplifier noise resistance in Ohms over frequency
        element_impedances (abc.Sequence[abc.Sequence[complex]]): A sequence of all elements,
            with the inner sequence representing the impedance of an element in Ohms over frequency

    Note: The frequencies for z_coil, z_out, fmin, rn and element_impedances are assumed to be the same

    Returns:
        abc.Sequence[float]: The noise figure in decibels over frequency
    """
    y_opt = 1 / np.array(z_out)
    matching_network_y_out = 1 / np.array(cls.calculate_output_impedance(z_coil, element_impedances)) # pylint: disable=no-value-for-parameter
    matching_network_gav_db = cls.calculate_available_power_gain(z_coil, element_impedances) # pylint: disable=no-value-for-parameter
    fmin_linear = 10 ** (np.array(fmin) / 10)
    preamp_nf_lin = fmin_linear + np.array(rn) / np.real(matching_network_y_out) * np.absolute(y_opt - matching_network_y_out) ** 2
    preamp_nf_db = 10 * np.log10(preamp_nf_lin)
    overall_nf_db = preamp_nf_db - matching_network_gav_db
    return overall_nf_db

class ExtendedBox1Topology(ReceiveCoilMatchingTopology):
  name = "Extended Box 1"

  netlist = """
Z1 1_1 3_1; down
Z2 1_1 5_0; right
Z4 5_0 4_0; down
Z3 3_1 4_0; right
Z5 4_0 0; right
W 5_0 2_0; right
W 1_0 1_1; right
W 3_0 3_1; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = 2*(-x11 * x22 + x12 ** 2) / (x12 - 2 * x22)
    x2 = x11 * x22 / x12 - x12
    x3 = x11 * (x11 * x22 - x12 ** 2) / x12 / (x11 - 2 * x12)
    x4 = (2 * x11 * x22 - 2 * x12 ** 2) / (x11 - 2 * x12)
    x5 = (-x11 * x22 + x12 ** 2) / (x11 - 2 * x12)
    return (x1, x2, x3, x4, x5)

class ExtendedBox2Topology(ReceiveCoilMatchingTopology):
  name = "Extended Box 2"
  
  netlist = """
Z2 5_0 4_0; down
Z3 5_0 2_1; right
Z5 2_1 0_1; down
Z4 4_0 0_1; right
Z1 3_0 4_0; right
W 1_0 5_0; right
W 2_1 2_0; right
W 0_1 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = (-x11 * x22 + x12 ** 2) / (x12 - 2 * x22)
    x2 = (-x11 * x22 + x12 ** 2) / (x12 - 2 * x22)
    x3 = x11 * x22 / 2 / x12 - x12 / 2
    x4 = (x11 * x22 - x12 ** 2) / (2 * x12 - 4 * x22)
    x5 = (x11 * x22 - x12 ** 2) / (x11 - 2 * x12)
    return (x1, x2, x3, x4, x5)

class ExtendedBox3Topology(ReceiveCoilMatchingTopology):
  name = "Extended Box 3"
  
  netlist = """
Z1 1_1 3_1; down
Z2 1_1 5_0; right
Z3 5_0 4_0; down
Z4 4_0 0_1; right
Z5 2_1 0_1; down
W 3_1 4_0; right
W 5_0 2_1; right
W 1_0 1_1; right
W 2_1 2_0; right
W 3_0 3_1; right
W 0_1 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = (2 * x12 ** 2 - 2 * x11 * x22) / (x12 - 2 * x22)
    x2 = x11 * x22 / x12 - x12
    x3 = x12 - x11 * x22 / x12
    x4 = x11 * x22 / 2 / x12 - x12 / 2
    x5 = x22 - x12 ** 2 / x11
    return (x1, x2, x3, x4, x5)

class ExtendedBox4Topology(ReceiveCoilMatchingTopology):
  name = "Extended Box 4"
  
  netlist = """
Z2 5_0 4_0; down
Z3 5_0 2_0; right
Z5 2_0 0_0; down
Z4 4_0 0_0; right
Z1 1_0 5_0; right
W 3_0 4_0; right
W 2_0 2_1; right
W 0_0 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = x11 * x22 / x12 - x12
    x2 = x12 - x11 * x22 / x12
    x3 = x22 - x12 / 2 + x11 * x22 * (x12 - 2 * x22) / 2 / x12 ** 2
    x4 = x11 * x22 / 2 / x12 - x12 / 2
    x5 = x22 - x12 ** 2 / x11
    return (x1, x2, x3, x4, x5)

class ExtendedLattice1Topology(ReceiveCoilMatchingTopology):
  name = "Extended Lattice 1"
  
  netlist = """
W 1_0 1_4; right
W 1_4 1_1; right
W 1_1 1_2; right
W 1_1 1_3; down
W 2_2 2_1; right
W 2_1 2_4; right
W 2_4 2_0; right
W 2_1 2_3; down
W 3_0 3_4; right
W 3_4 3_1; right
W 0_1 0_4; right
W 0_4 0; right
W 0_3 0_1; rotate=-45
W 3_3 3_1; rotate=225
Z1 1_3 0_3; right
Z2 1_2 2_2; right
Z3 3_1 0_1; right
Z4 3_3 2_3; right
Z5 2_4 0_4; down
W 0 0_5; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  has_only_one_solution = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = (-x11 * x22 + x12 ** 2) / (x12 - 2 * x22)
    x2 = x11 * x22 / x12 - x12
    x3 = (x11 * x22 - x12 ** 2) / (x12 + 2 * x22)
    x4 = x12 - x11 * x22 / x12
    x5 = x22 - x12 ** 2 / x11
    return (x1, x2, x3, x4, x5)

class ExtendedLattice2Topology(ReceiveCoilMatchingTopology):
  name = "Extended Lattice 2"
  
  netlist = """
Z2 1_1 2_0; right
W 2_0 2_2; right
W 2_0 2_1; right
Z4 2_1 3_2; rotate=225
W 3_2 3_1; rotate=225
W 3_1 3_0; right
Z3 3_0 4_3; right
Z1 1_2 4_0; rotate=-45
W 1_2 1_1; right
W 4_1 4_3; rotate=-45
W 4_0 4_1; right
W 4_3 4_2; right
W 2_2 2_3; right
Z5 4_2 0; right
W 1_0 1_2; right
W 3_3 3_1; right
W 0 0_5; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  has_only_one_solution = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = (2 * x12 ** 2 - 2 * x11 * x22) / (x11 - 4 * x22)
    x2 = 2 * (x11 * x22 - x12 ** 2) / (x11 + 2 * x12)
    x3 = (2 * x12 ** 2 - 2 * x11 * x22) / (x11 - 4 * x22)
    x4 = 2 * (x11 * x22 - x12 ** 2) / (x11 - 2 * x12)
    x5 = (x11 * x22 - x12 ** 2) / (x11 - 4 * x22)
    return (x1, x2, x3, x4, x5)

class ExtendedLattice3Topology(ReceiveCoilMatchingTopology):
  name = "Extended Lattice 3"
  
  netlist = """
Z3 1_1 2_0; right
W 2_0 2_2; right
W 2_0 2_1; right
Z5 2_1 4_2; rotate=225
W 4_2 4_1; rotate=225
W 4_1 4_0; right
Z4 4_0 0_3; right
Z2 1_2 0_0; rotate=-45
W 1_2 1_1; right
W 0_1 0_3; rotate=-45
W 0_0 0_1; right
W 0_3 0_2; right
W 2_2 2_3; right
W 0_2 0; right
W 1_0 1_2; right
Z1 3_0 4_1; right
W 0 0_5; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = x11 * (x11 * x22 - x12 ** 2) / (2 * x11 * x22 - x12 ** 2 - 2 * x12 * x22)
    x2 = (-x11 * x22 + x12 ** 2) / (x12 - 2 * x22)
    x3 = x11 * x22 / x12 - x12
    x4 = x12 * (x11 * x22 - x12 ** 2) / (2 * x12 * x22 + x12 ** 2 - 2 * x11 * x22)
    x5 = x12 + x11 * x22 * (x12 - 2 * x22) / (2 * x12 * x22 + x12 ** 2 - 2 * x11 * x22)
    return (x1, x2, x3, x4, x5)

class TroughTopology(ReceiveCoilMatchingTopology):
  name = "Trough"
  
  netlist = """
Z1 1_0 5_0; right
Z2 5_0 4_0; down
Z3 4_0 6_0; right
Z4 7_0 6_0; down
Z5 7_0 2_0; right
W 6_0 0; right
W 3_0 4_0; right
W 5_0 7_0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = x11 / 2 - x12
    x2 = x12 - x11 / 2
    x3 = x11 / 4 - x12 / 2
    x4 = x12 / 2 - x12 ** 2 / x11
    x5 = x22 - x12 / 2
    return (x1, x2, x3, x4, x5)

class CrestTopology(ReceiveCoilMatchingTopology):
  name = "Crest"
  
  netlist = """
W 1_0 5_0; right
Z2 5_0 4_0; down
W 4_0 6_0; right
Z4 7_0 6_0; down
W 7_0 2_0; right
Z5 6_0 0; right
Z1 3_0 4_0; right
Z3 5_0 7_0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = x11 / 2
    x2 = (2 * x12 ** 2 - 2 * x11 * x22) / (x12 - 2 * x22) - x11 / 2
    x3 = x11 / 4 + x11 * x22 / (2 * x12) - x12
    x4 = (x11 * x22 - x12 ** 2) / (x11 - 2 * x12) + x12 / 2
    x5 = -x12 / 2
    return (x1, x2, x3, x4, x5)

class HTopology(ReceiveCoilMatchingTopology):
  name = "H"
  
  netlist = """
Z1 1_0 5_0; right
Z3 5_0 4_0; down
Z2 3_0 4_0; right
Z4 5_0 2_0; right
Z5 4_0 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = True

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    x1 = x11 / 2 - x12
    x2 = x11 / 2
    x3 = x12
    x4 = x22 - x12 / 2
    x5 = -x12 / 2
    return (x1, x2, x3, x4, x5)

class PiTopology(ReceiveCoilMatchingTopology):
  name = "Pi"
  
  netlist = """
W 1_1 1_0; right
W 3_1 3_0; right
Z1 1_0 3_0; down
Z2 1_0 2_0; right
Z3 2_0 0_1; down
W 3_0 0_1; right
W 2_0 2_1; right
W 0_1 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 3

  has_common_mode_rejection = False

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    denominator = x12 ** 2 - x11 * x22
    b11 = x22 / denominator
    b12 = -x12 / denominator
    b22 = x11 / denominator
    b1 = b11 + b12
    b2 = -b12
    b3 = b22 + b12
    return (-1 / b1, -1 / b2, -1 / b3)

class TTopology(ReceiveCoilMatchingTopology):
  name = "T"
  
  netlist = """
Z1 1_0 5_0; right
Z2 5_0 4_0; down
Z3 5_0 2_0; right
W 3_0 4_0; right
W 4_0 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 3

  has_common_mode_rejection = False

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    return (x11 - x12, x12, x22 - x12)

class SymmetricBoxTopology(ReceiveCoilMatchingTopology):
  name = "Symmetric Box"
  
  netlist = """
Z1 1_0 3_0; down
Z2 1_0 2_0; right
Z4 2_0 0; down
Z3 3_0 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 4

  has_common_mode_rejection = False

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    pi_elements = PiTopology.calculate_elements_from_reactance_params(x11, x12, x22)
    return (pi_elements[0], pi_elements[1] / 2, pi_elements[1] / 2, pi_elements[2])

class SymmetricHTopology(ReceiveCoilMatchingTopology):
  name = "Symmetric H"
  
  netlist = """
Z1 1_0 5_0; right
Z3 5_0 4_0; down
Z4 5_0 2_0; right
Z2 3_0 4_0; right
Z5 4_0 0; right
W 0 0_4; down=0.1, ground
; label_nodes=none
"""

  num_elements = 5

  has_common_mode_rejection = False

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11, x12, x22):
    t_elements = TTopology.calculate_elements_from_reactance_params(x11, x12, x22)
    return (t_elements[0] / 2, t_elements[0] / 2, t_elements[1], t_elements[2] / 2, t_elements[2] / 2)
  