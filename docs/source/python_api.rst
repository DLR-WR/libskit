Python API Reference
====================
This section documents the Python API for LIBSkit, including both the core C++
classes exposed via pybind11 and the pure Python utility modules.

Core Classes
------------

These are the fundamental classes for plasma modeling and spectrum simulation.

Plasma Models
~~~~~~~~~~~~~

.. py:class:: libskit.LTEModel(element_ratio, temperature, electron_density, length)

   Local Thermodynamic Equilibrium (LTE) plasma model.

   This class models a plasma in local thermodynamic equilibrium, characterized
   by element compositions, temperature, electron density, and physical length.

   :param dict element_ratio: Dictionary mapping element symbols (str) to mass ratios (float).
                              Example: ``{"Fe": 0.7, "Si": 0.3}``
   :param float temperature: Plasma temperature in Kelvin (typically 8000-20000 K for LIBS)
   :param float electron_density: Electron number density in m⁻³ (typically 1e22 - 1e24 for LIBS)
   :param float length: Physical length of plasma column in meters

   .. py:attribute:: temperature
      :type: float

      Plasma temperature in Kelvin.

   .. py:attribute:: electron_density
      :type: float

      Electron number density in m⁻³.

   .. py:attribute:: length
      :type: float

      Physical length of plasma column in meters.

   .. py:attribute:: element_ratio
      :type: dict

      Dictionary of element mass ratios.

   .. py:method:: calculate_irradiance(wavelengths)

      Calculate the spectral irradiance of the plasma.

      :param numpy.ndarray wavelengths: Wavelength array in nanometers
      :return: Irradiance values at each wavelength
      :rtype: numpy.ndarray

   .. py:method:: get_lines()

      Get emission lines with ion, wavelength, peak emission, and FWHM.

      :return: List of emission line data
      :rtype: list

   **Example:**

   .. code-block:: python

      import libskit
      import numpy as np

      model = libskit.LTEModel(
          {"Fe": 0.7, "Ca": 0.3},
          temperature=10000.0,
          electron_density=1e23,
          length=1e-3
      )

      wavelengths = np.linspace(200, 900, 1000)
      irradiance = model.calculate_irradiance(wavelengths)


.. py:class:: libskit.LTEModelScaled(element_ratio, normalized_temperature, normalized_electron_density, normalized_length, temperature_scale=10000.0, electron_density_scale=1e24, length_scale=1e-3)

   Scaled parameter version of LTEModel for gradient descent optimization.

   This model uses normalized parameters (typically 0-1 range) that are scaled
   to physical values using the provided scale factors. This improves numerical
   stability during optimization.

   :param dict element_ratio: Dictionary mapping element symbols to mass ratios
   :param float normalized_temperature: Normalized temperature value
   :param float normalized_electron_density: Normalized electron density value
   :param float normalized_length: Normalized length value
   :param float temperature_scale: Scale factor for temperature (default: 10000.0)
   :param float electron_density_scale: Scale factor for electron density (default: 1e24)
   :param float length_scale: Scale factor for length (default: 1e-3)


.. py:class:: libskit.LTELayers(element_ratios, temperatures, electron_densities, lengths)

   Multi-layer plasma model for modeling spatially varying plasmas.

   This class represents a plasma with multiple layers, each having its own
   elemental composition and plasma conditions. Useful for modeling
   inhomogeneous plasmas or self-absorption effects.

   :param list element_ratios: List of element ratio dictionaries, one per layer
   :param list temperatures: List of temperatures in Kelvin, one per layer
   :param list electron_densities: List of electron densities in m⁻³, one per layer
   :param list lengths: List of physical lengths in meters, one per layer

   **Example:**

   .. code-block:: python

      import libskit

      # Two-layer plasma with different conditions
      layers = libskit.LTELayers(
          [{"Fe": 1.0}, {"Fe": 1.0}],           # Element ratios
          [12000.0, 8000.0],                     # Temperatures
          [1e24, 1e23],                          # Electron densities
          [1e-4, 1e-3]                           # Lengths
      )


.. py:class:: libskit.LTETimeIntegrated(element_ratio, start_temperature, end_temperature, tau_temperature, start_electron_density, end_electron_density, tau_electron_density, start_length, end_length, tau_length, integration_time, integration_points=5)

   Time-integrated plasma model.

   Models the time evolution of a LIBS plasma where temperature and
   electron density decay asymptotically toward final values, and
   the plasma length grows with saturation.

   :param dict element_ratio: Dictionary mapping element symbols to mass ratios
   :param float start_temperature: Initial temperature T_0 in Kelvin
   :param float end_temperature: Asymptotic final temperature T_f in Kelvin
   :param float tau_temperature: Temperature decay time constant in seconds
   :param float start_electron_density: Initial electron density ne_0 in m⁻³
   :param float end_electron_density: Asymptotic final electron density ne_f in m⁻³
   :param float tau_electron_density: Electron density decay time constant
   :param float start_length: Initial plasma length L_0 in meters
   :param float end_length: Asymptotic final plasma length L_f in meters
   :param float tau_length: Length evolution time constant
   :param float integration_time: Total integration time in seconds
   :param int integration_points: Number of time points for the weighted sum (default: 5)


Instruments
~~~~~~~~~~~

.. py:class:: libskit.GenericSpectrometer(sigma, gamma)

   Generic spectrometer model with configurable Voigt line broadening.

   This class simulates how a spectrometer would measure plasma emission,
   applying instrumental broadening using a Voigt profile (combination of
   Gaussian and Lorentzian).

   :param float sigma: Gaussian broadening parameter (standard deviation) in nm
   :param float gamma: Lorentzian broadening parameter (HWHM) in nm

   .. py:attribute:: sigma
      :type: float

      Gaussian broadening parameter in nm.

   .. py:attribute:: gamma
      :type: float

      Lorentzian broadening parameter in nm.

   .. py:method:: calculate_spectrum(wavelengths, plasma_model)

      Calculate the measured spectrum with instrumental broadening applied.

      :param numpy.ndarray wavelengths: Wavelength array in nanometers
      :param PlasmaModel plasma_model: Plasma model to simulate
      :return: Broadened spectrum intensities
      :rtype: numpy.ndarray

   **Example:**

   .. code-block:: python

      import libskit
      import numpy as np

      instrument = libskit.GenericSpectrometer(sigma=0.05, gamma=0.01)
      model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)

      wavelengths = np.linspace(200, 900, 5000)
      spectrum = instrument.calculate_spectrum(wavelengths, model)


.. py:class:: libskit.Aryelle()

   Aryelle spectrometer model with instrument-specific characteristics.

   This class implements the specific instrumental response function of the
   Aryelle spectrometer system.

   .. py:method:: calculate_spectrum(wavelengths, plasma_model)

      Calculate the measured spectrum using Aryelle instrumental characteristics.

      :param numpy.ndarray wavelengths: Wavelength array in nanometers
      :param PlasmaModel plasma_model: Plasma model to simulate
      :return: Broadened spectrum intensities
      :rtype: numpy.ndarray


Data Structures
~~~~~~~~~~~~~~~

.. py:class:: libskit.Spectrum(wavelengths, irradiance)

   Container for spectroscopic data with wavelength-intensity pairs.

   :param numpy.ndarray wavelengths: Wavelength values in nanometers
   :param numpy.ndarray irradiance: Corresponding intensity values

   .. py:attribute:: wavelengths
      :type: numpy.ndarray

      Wavelength array in nanometers.

   .. py:attribute:: irradiance
      :type: numpy.ndarray

      Intensity/irradiance array.


.. py:class:: libskit.Experiment(loss_type, plasma_model, instrument, true_spectrum)

   Experimental configuration for gradient-based optimization.

   Combines all components needed for computing gradients of loss functions
   with respect to plasma model parameters.

   :param LossType loss_type: Type of loss function (L2Loss or IntensityWeightedLoss)
   :param PlasmaModel plasma_model: Initial plasma model
   :param Instrument instrument: Spectrometer model
   :param Spectrum true_spectrum: Target experimental spectrum


.. py:class:: libskit.LossType

   Enumeration of available loss function types.

   .. py:attribute:: L2Loss

      Standard L2 (least squares) loss function.

   .. py:attribute:: IntensityWeightedLoss

      Custom loss function designed for LIBS data.


Core Functions
~~~~~~~~~~~~~~

.. py:function:: libskit.calculate_partition_function(ion, temperature)

   Calculate the partition function for a given ion and temperature.

   Uses the full energy level data from the NIST database.

   :param str ion: Ion species string, e.g., "Ca I", "Fe II"
   :param float temperature: Temperature in Kelvin
   :return: Partition function value
   :rtype: float

   **Example:**

   .. code-block:: python

      >>> libskit.calculate_partition_function("Ca I", 10000.0)
      1.234


.. py:function:: libskit.calculate_saha_function(element, ionization_level, temperature)

   Calculate the Saha function for a given element and ionization level.

   :param str element: Element symbol, e.g., "Ca", "Fe"
   :param int ionization_level: Ionization level (0 = neutral to singly ionized)
   :param float temperature: Temperature in Kelvin
   :return: Saha function value in m⁻³
   :rtype: float


.. py:function:: libskit.get_gradients(experiment)

   Calculate gradients for all parameters in an experiment.

   Uses automatic differentiation to compute gradients of the loss function
   with respect to all plasma model parameters.

   :param Experiment experiment: Experiment configuration
   :return: Dictionary mapping parameter names to gradient values
   :rtype: dict


.. py:function:: libskit.loss(experiment)

   Calculate the loss function value for an experiment.

   :param Experiment experiment: Experiment configuration
   :return: Computed loss value
   :rtype: float


Database Management
~~~~~~~~~~~~~~~~~~~

.. py:function:: libskit.set_database_path(path)

   Set the path to the atomic databases directory.

   :param str path: Path to directory containing CSV database files

.. py:function:: libskit.get_database_path()

   Get the current path to the atomic databases directory.

   :return: Current database path
   :rtype: str

.. py:function:: libskit.load_databases()

   Manually trigger loading of atomic databases.

.. py:function:: libskit.reload_databases()

   Force reload of atomic databases from current database path.


Accuracy Settings
~~~~~~~~~~~~~~~~~

.. py:function:: libskit.set_emission_accuracy(accuracy)

   Set the emission line filtering accuracy threshold.

   Controls how weak emission lines are filtered during spectrum calculation.

   :param float accuracy: Threshold value (default: 1e-6)

.. py:function:: libskit.get_emission_accuracy()

   Get the current emission line filtering accuracy threshold.

   :return: Current emission accuracy threshold
   :rtype: float

.. py:function:: libskit.set_broadening_accuracy(accuracy)

   Set the spectral line broadening accuracy threshold.

   :param float accuracy: Threshold value (default: 1e-5)

.. py:function:: libskit.get_broadening_accuracy()

   Get the current spectral line broadening accuracy threshold.

   :return: Current broadening accuracy threshold
   :rtype: float


Python Utility Modules
----------------------

Boltzmann Plot Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: libskit.boltzmann_plot
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: libskit.boltzmann_utils
   :members:
   :undoc-members:
   :show-inheritance:

Spectral Analysis
~~~~~~~~~~~~~~~~~

.. automodule:: libskit.spectral_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Descent Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: libskit.gradient_descent
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
~~~~~~~~~~~~~

.. automodule:: libskit.visualization
   :members:
   :undoc-members:
   :show-inheritance:

File Parsers
~~~~~~~~~~~~

.. automodule:: libskit.parsers
   :members:
   :undoc-members:
   :show-inheritance:

Database Generation
~~~~~~~~~~~~~~~~~~~

.. automodule:: libskit.generate_databases
   :members:
   :undoc-members:
   :show-inheritance:

Convergence Testing
~~~~~~~~~~~~~~~~~~~

.. automodule:: libskit.convergence_test
   :members:
   :undoc-members:
   :show-inheritance:

Database Verification
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: libskit.database_verification
   :members:
   :undoc-members:
   :show-inheritance:
