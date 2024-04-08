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
from collections import abc
import numpy as np
import skrf

class ReceiveCoilMatchingTopology(lc_power_match_baluns.topology.BalunTopology):
  """A matching network topology for an MR receive coil and a preamplifier"""

  has_common_mode_rejection: bool = False

  @classmethod
  def calculate_elements_from_reactance_params(cls, x11: float, x12: float,
      x22: float) -> tuple[float, ...]:
    """Calculates element reactances from reactance parameters for optimal matching/decoupling/common-mode rejection

    Args:
        x11 (float): X11 reactance parameter
        x12 (float): X12 reactance parameter
        x22 (float): X22 reactance parameter

    Raises:
        NotImplementedError: If the topology has not implemented this method

    Returns:
        tuple[float, ...]: The element reactances
    """
    raise NotImplementedError
  
  @classmethod
  def calculate_preamplifier_decoupling(cls, rcoil: float, xcoil: float,
      ramp: float, xamp: float, element_impedances: abc.Sequence[complex]) -> float:
    """Calculates preamplifier decoupling from element impedances using the measure described by [1]
    [1] W. Wang, V. Zhurbenko, J. D. Sánchez‐Heredia, and J. H. Ardenkjær‐Larsen, "Trade‐off between preamplifier noise figure and decoupling in MRI detectors," Magnetic Resonance in Medicine, vol. 89, no. 2, pp. 859–871, 2023. doi:10.1002/mrm.29489 

    Args:
        rcoil (float): The coil resistance
        xcoil (float): The coil reactance
        ramp (float): The input resistance of the preamplifier
        xamp (float): The input reactance of the preamplifier
        element_impedances (abc.Sequence[complex]): The impedances of each element

    Returns:
        float: The preamplifier decoupling in decibels
    """
    twoport_s = cls.calculate_two_port_scattering_parameters(rcoil, xcoil, ramp, xamp, element_impedances, "power")
    s11 = twoport_s[0, 0, 0]
    decoupling = -20 * np.log10(np.abs(1 - s11))
    return decoupling
  
  @classmethod
  def calculate_noise_figure(cls, rcoil: float, xcoil: float, rout: float,
    xout: float, fmin: float, rn: float, element_impedances: abc.Sequence[complex]) -> float:
    """Calculates a noise figure estimate from element impedances

    Args:
        rcoil (float): The coil resistance
        xcoil (float): The coil reactance
        rout (float): The output resistance of the matching network
        xout (float): The output reactance of the matching network
        fmin (float): The minimum noise figure for the preamplifier in decibels
        rn (float): The preamplifier noise resistance
        element_impedances (abc.Sequence[complex]): The impedances of each element

    Returns:
        float: The noise figure in decibels
    """
    yopt = 1 / (rout + 1j * xout)
    zcoil = rcoil + 1j * xcoil
    ycoil = 1 / zcoil
    coil_z = np.array([[[zcoil]]])
    coil_network = skrf.Network.from_z(coil_z)
    twoport_z = cls.calculate_two_port_impedance_parameters(element_impedances)
    matching_network = skrf.Network.from_z(twoport_z)
    loaded_matching_network = skrf.network.connect(coil_network, 0, matching_network, 0)
    matching_network_yout = loaded_matching_network.y[0, 0, 0]
    matching_network_y = matching_network.y
    matching_network_gav = np.absolute(matching_network_y[0, 1, 0] / (matching_network_y[0, 0, 0] + ycoil)) ** 2 * np.real(ycoil) / np.real(matching_network_yout)
    fmin_linear = 10 ** (fmin / 10)
    preamp_nf = fmin_linear + rn / np.real(matching_network_yout) * np.absolute(yopt - matching_network_yout) ** 2
    overall_nf = preamp_nf / matching_network_gav
    overall_nf_db = 10 * np.log10(overall_nf)
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
Z2 1_2 2_0; right
W 2_0 2_4; right
Z3 2_2 3_3; rotate=225
W 3_3 3_0; rotate=225
Z4 3_0 0_3; right
W 1_0 1_2; right
W 0_3 0_2; right
Z1 1_0 3_1; down
W 3_1 3_0; right
W 2_4 2_1; right
Z5 2_1 0_2; down
W 2_4 2_2; down
W 1_1 1_0; right
W 3_2 3_1; right
W 2_1 2_3; right
W 0_2 0; right
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

class TildeTopology(ReceiveCoilMatchingTopology):
  name = "Tilde"
  
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
    x1 = 2 * (x12 ** 2 - x11 * x22) / (x12 - 2 * x22)
    x2 = x11 * x22 / x12 - x12
    x3 = x12 - x11 * x22 / x12
    x4 = x11 * x22 / 2 / x12 - x12 / 2
    x5 = x22 - x12 ** 2 / x11
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
    denom = x12 ** 2 - x11 * x22
    b11 = x22 / denom
    b12 = -x12 / denom
    b22 = x11 / denom
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
  