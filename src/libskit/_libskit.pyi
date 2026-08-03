"""

        LIBSkit C++ Extension Module
        ============================

        This module provides the core C++ functionality for LIBSkit, a library
        for simulating and analyzing Laser-Induced Breakdown Spectroscopy (LIBS)
        spectra.

        Example
        -------
        >>> import libskit
        >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
        >>> instrument = libskit.GenericSpectrometer(0.05, 0.01)
        >>> wavelengths = np.linspace(200, 900, 5000)
        >>> spectrum = instrument.calculate_spectrum(wavelengths, model)
    
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AnalyticNorm', 'Aryelle', 'ChemCam', 'ENZYME_ENABLED', 'Experiment', 'GenericSpectrometer', 'Instrument', 'L1Loss', 'L2Loss', 'LTELayers', 'LTEModel', 'LTEModelScaled', 'LTETimeIntegrated', 'LogL2Loss', 'LossType', 'MaxNorm', 'NormalizationType', 'OptimizedScaleNorm', 'IntensityWeightedLoss', 'PlasmaModel', 'SlopingSpectrometer', 'Spectrum', 'WeightedL2Loss', 'blackbody_irradiance', 'calculate_irradiance_test', 'calculate_partition_function', 'calculate_saha_function', 'compute_loss_enzyme', 'get_accuracy', 'get_aki_accuracies', 'get_database_path', 'get_fine_samples_per_fwhm', 'get_gradients_enzyme', 'get_gradients_forward', 'load_databases', 'loss', 'optimal_scale_factor', 'reload_databases', 'reset_aki', 'set_accuracy', 'set_aki_multipliers', 'set_database_path', 'set_fine_samples_per_fwhm']
class Aryelle(Instrument):
    """
    
              Aryelle spectrometer model with instrument-specific characteristics.
    
              This class implements the specific instrumental response function of
              the LTB Aryelle spectrometer.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>> aryelle = libskit.Aryelle()
              >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
              >>> wavelengths = np.linspace(200, 900, 5000)
              >>> spectrum = aryelle.calculate_spectrum(wavelengths, model)
          
    """
    def __init__(self) -> None:
        ...
    def calculate_spectrum(self, wavelengths: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], model: PlasmaModel) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
class ChemCam(Instrument):
    """
    
              ChemCam spectrometer model for the Mars Science Laboratory rover.
    
              Models the ChemCam LIBS instrument which consists of three separate
              CCD spectrometers with different spectral resolutions:
    
              - UV:     240.1-342.2 nm,  FWHM = 0.15 nm
              - VIO:    382.1-469.3 nm,  FWHM = 0.20 nm
              - VISNIR: 474.0-906.5 nm,  FWHM = 0.61 nm
    
              Instrumental broadening is a pure Gaussian for each range.
              Wavelengths in the inter-detector gaps (342.2-382.1 nm, 469.3-474.0 nm)
              receive zero intensity.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>> chemcam = libskit.ChemCam()
              >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
              >>> wavelengths = np.linspace(240, 840, 10000)
              >>> spectrum = chemcam.calculate_spectrum(wavelengths, model)
          
    """
    def __init__(self) -> None:
        ...
    def calculate_spectrum(self, wavelengths: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], model: PlasmaModel) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    @property
    def scale_vio(self) -> float:
        """
        Relative scale factor for the VIO CCD (382.1–469.3 nm). UV is the reference (implicitly 1.0).
        """
    @scale_vio.setter
    def scale_vio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def scale_visnir(self) -> float:
        """
        Relative scale factor for the VISNIR CCD (474.0–906.5 nm). UV is the reference (implicitly 1.0).
        """
    @scale_visnir.setter
    def scale_visnir(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Experiment:
    """
    
              Experimental configuration for gradient-based optimization.
    
              This class encapsulates all components needed for computing gradients
              of loss functions with respect to plasma model parameters. It combines:
              - The plasma model (with parameters to be optimized)
              - The instrument model (for forward simulation)
              - The experimental data (target spectrum)
              - The loss function type
    
              Parameters
              ----------
              loss_type : LossType
                  Type of loss function to use (L2Loss or IntensityWeightedLoss).
              plasma_model : PlasmaModel
                  Initial plasma model with parameters to optimize.
              instrument : Instrument
                  Spectrometer model for simulating measurements.
              true_spectrum : Spectrum
                  Target experimental spectrum to fit.
    
              Attributes
              ----------
              loss_type : LossType
                  The loss function type.
              plasma_model : PlasmaModel
                  The plasma model being optimized.
              instrument : Instrument
                  The instrument model.
              true_spectrum : Spectrum
                  The target spectrum.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>>
              >>> # Create components
              >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
              >>> instrument = libskit.GenericSpectrometer(0.05, 0.01)
              >>> wavelengths = np.linspace(200, 900, 5000)
              >>> intensities = instrument.calculate_spectrum(wavelengths, model)
              >>> spectrum = libskit.Spectrum(wavelengths, intensities)
              >>>
              >>> # Create experiment
              >>> experiment = libskit.Experiment(
              ...     libskit.LossType.L2Loss,
              ...     model,
              ...     instrument,
              ...     spectrum
              ... )
              >>>
              >>> # Calculate loss and gradients
              >>> loss_value = libskit.loss(experiment)
              >>> gradients = libskit.get_gradients_enzyme(experiment)
          
    """
    instrument: Instrument
    loss_type: LossType
    normalization_type: NormalizationType
    plasma_model: PlasmaModel
    true_spectrum: Spectrum
    def __init__(self, loss_type: LossType, plasma_model: PlasmaModel, instrument: Instrument, true_spectrum: Spectrum, normalization_type: NormalizationType = ...) -> None:
        ...
    def get_instrument(self) -> Instrument:
        ...
    def get_plasma_model(self) -> PlasmaModel:
        ...
    @property
    def loss_weights(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Per-pixel weights for WeightedL2Loss (length = spectrum length, or empty for uniform weights).
        """
    @loss_weights.setter
    def loss_weights(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    @property
    def scale_factor(self) -> float:
        ...
    @scale_factor.setter
    def scale_factor(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class GenericSpectrometer(Instrument):
    """
    
              Generic spectrometer model with configurable Voigt line broadening.
    
              This class simulates how a spectrometer would measure plasma emission,
              applying instrumental broadening using a Voigt profile (combination of
              Gaussian and Lorentzian line shapes). The broadening is performed
              efficiently using FFT convolution.
    
              Parameters
              ----------
              sigma : float
                  Gaussian broadening parameter (standard deviation) in nanometers.
                  Represents the Doppler-like instrumental broadening component.
              gamma : float
                  Lorentzian broadening parameter (half-width at half-maximum) in
                  nanometers. Represents pressure-like broadening effects.
    
              Attributes
              ----------
              sigma : float
                  Gaussian broadening parameter in nm.
              gamma : float
                  Lorentzian broadening parameter in nm.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>> instrument = libskit.GenericSpectrometer(sigma=0.05, gamma=0.01)
              >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
              >>> wavelengths = np.linspace(200, 900, 5000)
              >>> spectrum = instrument.calculate_spectrum(wavelengths, model)
          
    """
    def __init__(self, sigma: typing.SupportsFloat | typing.SupportsIndex, gamma: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def calculate_spectrum(self, wavelengths: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], model: PlasmaModel) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sigma(self) -> float:
        ...
    @sigma.setter
    def sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Instrument:
    """
    
              Abstract base class for spectroscopic instrument models.
    
              This class defines the interface for all instrument models in LIBSkit.
              Instruments simulate how plasma spectra would be measured by real
              spectrometers, including effects such as instrumental line broadening.
          
    """
class LTELayers(PlasmaModel):
    """
    
              Multi-layer plasma model for spatially varying plasmas.
    
              This class represents a plasma with multiple discrete layers, each
              having its own elemental composition, temperature, electron density,
              and length. It is useful for modeling:
              - Inhomogeneous plasmas with temperature gradients
              - Core-shell plasma structures
    
              Parameters
              ----------
              element_ratios : list of dict
                  List of element ratio dictionaries, one per layer.
              temperatures : list of float
                  List of temperatures in Kelvin, one per layer.
              electron_densities : list of float
                  List of electron densities in m^-3, one per layer.
              lengths : list of float
                  List of physical lengths in meters, one per layer.
    
              Example
              -------
              >>> # Two-layer plasma with different conditions
              >>> layers = libskit.LTELayers(
              ...     [{"Fe": 1.0}, {"Fe": 1.0}],  # Element ratios per layer
              ...     [12000.0, 8000.0],             # Temperatures (K)
              ...     [1e24, 1e23],                  # Electron densities (m^-3)
              ...     [1e-4, 1e-3]                   # Lengths (m)
              ... )
              >>> len(layers)  # Number of layers
              2
              >>> layers[0].temperature  # Access first layer
              12000.0
          
    """
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> LTEModel:
        ...
    def __init__(self, arg0: list, arg1: list, arg2: list, arg3: list) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def calculate_irradiance(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_parameter(self, arg0: str) -> float:
        ...
    def get_parameter_names(self) -> list[str]:
        ...
    def set_parameter(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class LTEModel(PlasmaModel):
    """
    
              Local Thermodynamic Equilibrium (LTE) plasma model.
    
              This class models a plasma in local thermodynamic equilibrium,
              characterized by its element mass ratios, temperature, electron
              density, and physical length. It provides methods for calculating
              population fractions, emission line intensities, and radiative
              transfer.
    
              Parameters
              ----------
              element_ratio : dict
                  Dictionary mapping element symbols (str) to mass ratios (float).
                  The ratios should sum to 1.0 for proper normalization.
                  Example: {"Fe": 0.7, "Si": 0.2, "Ca": 0.1}
              temperature : float
                  Plasma temperature in Kelvin. Typical LIBS values range from
                  8,000 K to 20,000 K.
              electron_density : float
                  Electron number density in m^-3. Typical LIBS values range from
                  1e22 to 1e24 m^-3.
              length : float
                  Physical length of the plasma column in meters. Typical LIBS
                  values range from 1e-4 to 1e-2 m.
    
              Attributes
              ----------
              temperature : float
                  Plasma temperature in Kelvin.
              electron_density : float
                  Electron number density in m^-3.
              length : float
                  Physical length of plasma column in meters.
              element_ratio : dict
                  Dictionary of element mass ratios.
    
              Methods
              -------
              calculate_irradiance(wavelengths)
                  Calculate spectral irradiance at given wavelengths.
              get_lines()
                  Get emission lines with wavelength, intensity, and FWHM.
              calculate_element_fractions()
                  Calculate element number density fractions.
              get_population_fractions()
                  Get ionization state populations for each element.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>> model = libskit.LTEModel(
              ...     {"Fe": 0.7, "Ca": 0.3},
              ...     temperature=10000.0,
              ...     electron_density=1e23,
              ...     length=1e-3
              ... )
              >>> wavelengths = np.linspace(200, 900, 1000)
              >>> irradiance = model.calculate_irradiance(wavelengths)
              >>> print(f"Temperature: {model.temperature} K")
              Temperature: 10000.0 K
    
              Notes
              -----
              The LTE assumption requires that the plasma is sufficiently dense
              and collisionally dominated. The model uses the Saha-Boltzmann
              equations to calculate ionization equilibrium and level populations.
          
    """
    def __contains__(self, arg0: str) -> bool:
        ...
    def __getitem__(self, arg0: str) -> float:
        ...
    def __init__(self, arg0: collections.abc.Mapping[str, typing.SupportsFloat | typing.SupportsIndex], arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def calculate_continuum_emission(self, wavelengths: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
                  Calculate free-free and free-bound continuum emission separately.
        
                  Computes bremsstrahlung (free-free) and recombination (free-bound)
                  continuum emission in the hydrogen approximation (Gaunt factor = 1).
                  Reference: Noll (2012), Eqs. 10.14-10.18.
        
                  Parameters
                  ----------
                  wavelengths : numpy.ndarray
                      Wavelength grid in nanometers.
        
                  Returns
                  -------
                  tuple[numpy.ndarray, numpy.ndarray]
                      A tuple (free_free, free_bound), each an array of emission
                      coefficients in W/m⁴ at the given wavelengths.
        
                  Example
                  -------
                  >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
                  >>> ff, fb = model.calculate_continuum_emission(wavelengths)
        """
    def calculate_element_fractions(self) -> dict[str, float]:
        ...
    def calculate_irradiance(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_lines(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> list[tuple[str, float, float, float, float]]:
        """
        Get emission lines with ion, wavelength, peak emission, FWHM, and accuracy fraction
        """
    def get_parameter(self, arg0: str) -> float:
        """
        Get a parameter value by name
        """
    def get_parameter_names(self) -> list[str]:
        """
        Get list of all parameter names (temperature, electron_density, length, elements)
        """
    def get_population_fractions(self) -> dict[str, typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        ...
    def items(self) -> list:
        ...
    def keys(self) -> list:
        ...
    def set_element_ratio(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the ratio for a specific element
        """
    def set_parameter(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set a parameter value by name
        """
    @property
    def electron_density(self) -> float:
        ...
    @electron_density.setter
    def electron_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def element_ratio(self) -> dict[str, float]:
        ...
    @element_ratio.setter
    def element_ratio(self, arg1: collections.abc.Mapping[str, typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @property
    def include_continuum(self) -> bool:
        """
                  Whether to include continuum emission (bremsstrahlung + recombination).
        
                  When True, free-free (bremsstrahlung) and free-bound (recombination)
                  continuum radiation is added to the emission coefficient before solving
                  the radiative transfer equation. Uses the hydrogen approximation
                  (Gaunt factor = 1).
        
                  Default is False for backward compatibility.
        
                  Example
                  -------
                  >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
                  >>> model.include_continuum = True
                  >>> irradiance = model.calculate_irradiance(wavelengths)
        """
    @include_continuum.setter
    def include_continuum(self, arg0: bool) -> None:
        ...
    @property
    def length(self) -> float:
        ...
    @length.setter
    def length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def temperature(self) -> float:
        ...
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class LTEModelScaled(LTEModel):
    """
    
              Scaled parameter version of LTEModel for gradient descent optimization.
    
              This model uses normalized parameters (typically 0-1 range) that are
              scaled to physical values using the provided scale factors. This
              improves numerical stability during gradient-based optimization by
              ensuring all parameters have similar magnitudes.
    
              Parameters
              ----------
              element_ratio : dict
                  Dictionary mapping element symbols to mass ratios.
              normalized_temperature : float
                  Normalized temperature value (dimensionless).
              normalized_electron_density : float
                  Normalized electron density value (dimensionless).
              normalized_length : float
                  Normalized length value (dimensionless).
              temperature_scale : float, optional
                  Scale factor for temperature (default: 10000.0 K).
              electron_density_scale : float, optional
                  Scale factor for electron density (default: 1e24 m^-3).
              length_scale : float, optional
                  Scale factor for length (default: 1e-3 m).
    
              Notes
              -----
              The actual physical values are calculated as:
              - temperature = normalized_temperature * temperature_scale
              - electron_density = normalized_electron_density * electron_density_scale
              - length = normalized_length * length_scale
    
              Example
              -------
              >>> model = libskit.LTEModelScaled(
              ...     {"Fe": 1.0},
              ...     normalized_temperature=1.0,      # = 10000 K
              ...     normalized_electron_density=0.1, # = 1e23 m^-3
              ...     normalized_length=1.0            # = 1e-3 m
              ... )
          
    """
    def __init__(self, element_ratio: collections.abc.Mapping[str, typing.SupportsFloat | typing.SupportsIndex], normalized_temperature: typing.SupportsFloat | typing.SupportsIndex, normalized_electron_density: typing.SupportsFloat | typing.SupportsIndex, normalized_length: typing.SupportsFloat | typing.SupportsIndex, temperature_scale: typing.SupportsFloat | typing.SupportsIndex = 10000.0, electron_density_scale: typing.SupportsFloat | typing.SupportsIndex = 1e+24, length_scale: typing.SupportsFloat | typing.SupportsIndex = 0.001) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_normalized_parameter(self, arg0: str) -> float:
        ...
    def set_normalized_parameter(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def electron_density_scale(self) -> float:
        ...
    @electron_density_scale.setter
    def electron_density_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def length_scale(self) -> float:
        ...
    @length_scale.setter
    def length_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def normalized_electron_density(self) -> float:
        ...
    @normalized_electron_density.setter
    def normalized_electron_density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def normalized_length(self) -> float:
        ...
    @normalized_length.setter
    def normalized_length(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def normalized_temperature(self) -> float:
        ...
    @normalized_temperature.setter
    def normalized_temperature(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def temperature_scale(self) -> float:
        ...
    @temperature_scale.setter
    def temperature_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class LTETimeIntegrated(PlasmaModel):
    """
    
              Time-integrated plasma model.
    
              Models the time evolution of a LIBS plasma where temperature
              and electron density decay asymptotically toward final values,
              and the plasma length grows with saturation:
    
                T(t)  = T_f  + (T_0  - T_f)  * exp(-t / tau_T)
                ne(t) = ne_f + (ne_0 - ne_f) * exp(-t / tau_ne)
                L(t)  = L_f  + (L_0  - L_f)  * exp(-t / tau_L)
    
              The resulting spectrum is integrated over the specified time window.
    
              Parameters
              ----------
              element_ratio : dict
                  Dictionary mapping element symbols to mass ratios.
              start_temperature : float
                  Initial temperature T_0 at t=0 in Kelvin.
              end_temperature : float
                  Asymptotic final temperature T_f in Kelvin.
              tau_temperature : float
                  Temperature decay time constant in seconds.
              start_electron_density : float
                  Initial electron density ne_0 at t=0 in m^-3.
              end_electron_density : float
                  Asymptotic final electron density ne_f in m^-3.
              tau_electron_density : float
                  Electron density decay time constant in seconds.
              start_length : float
                  Initial plasma length L_0 at t=0 in meters.
              end_length : float
                  Asymptotic final plasma length L_f in meters.
              tau_length : float
                  Length evolution time constant in seconds.
              integration_time : float
                  Total integration time in seconds.
              integration_points : int, optional
                  Number of time points for the weighted sum (default: 5).
    
              Example
              -------
              >>> model = libskit.LTETimeIntegrated(
              ...     {"Fe": 1.0},
              ...     start_temperature=13000.0,
              ...     end_temperature=6500.0,
              ...     tau_temperature=195e-9,
              ...     start_electron_density=4e23,
              ...     end_electron_density=0.8e23,
              ...     tau_electron_density=90e-9,
              ...     start_length=0.5e-3,
              ...     end_length=3.0e-3,
              ...     tau_length=109e-9,
              ...     integration_time=1990e-9,
              ...     integration_points=10
              ... )
          
    """
    def __init__(self, element_ratio: collections.abc.Mapping[str, typing.SupportsFloat | typing.SupportsIndex], start_temperature: typing.SupportsFloat | typing.SupportsIndex, end_temperature: typing.SupportsFloat | typing.SupportsIndex, tau_temperature: typing.SupportsFloat | typing.SupportsIndex, start_electron_density: typing.SupportsFloat | typing.SupportsIndex, end_electron_density: typing.SupportsFloat | typing.SupportsIndex, tau_electron_density: typing.SupportsFloat | typing.SupportsIndex, start_length: typing.SupportsFloat | typing.SupportsIndex, end_length: typing.SupportsFloat | typing.SupportsIndex, tau_length: typing.SupportsFloat | typing.SupportsIndex, integration_time: typing.SupportsFloat | typing.SupportsIndex, integration_points: typing.SupportsInt | typing.SupportsIndex = 5) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def calculate_irradiance(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_parameter(self, arg0: str) -> float:
        ...
    def get_parameter_names(self) -> list[str]:
        ...
    def set_parameter(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def end_electron_density(self) -> float:
        ...
    @property
    def end_length(self) -> float:
        ...
    @property
    def end_temperature(self) -> float:
        ...
    @property
    def integration_points(self) -> int:
        ...
    @property
    def integration_time(self) -> float:
        ...
    @property
    def start_electron_density(self) -> float:
        ...
    @property
    def start_length(self) -> float:
        ...
    @property
    def start_temperature(self) -> float:
        ...
    @property
    def tau_electron_density(self) -> float:
        ...
    @property
    def tau_length(self) -> float:
        ...
    @property
    def tau_temperature(self) -> float:
        ...
class LossType:
    """
    
              Enumeration of available loss function types for optimization.
    
              Different loss functions can be used depending on the characteristics
              of the data and the desired optimization behavior.
    
              Attributes
              ----------
              L2Loss
                  Standard L2 (least squares) loss function. Minimizes the sum of
                  squared residuals between measured and simulated spectra.
              IntensityWeightedLoss
                  Custom loss function designed for LIBS data that is more robust
                  to outliers and baseline variations.
              L1Loss
                  L1 (sum of absolute residuals) loss. More robust to outliers
                  than L2 — large residuals are penalised linearly, not
                  quadratically.
              LogL2Loss
                  L2 loss applied to log(1 + spectrum) transformed data.
                  The logarithmic compression amplifies the contribution of
                  small peaks relative to dominant ones, improving recovery
                  of minor/trace species.
              WeightedL2Loss
                  Per-pixel weighted L2 loss
                  Σ w_i (y_sim_i - y_exp_i)² / Σ w_i.
                  The weight vector is set on the Experiment via the
                  ``loss_weights`` attribute (length must equal the
                  spectrum length). Useful for downweighting spectral
                  regions with high transition-probability uncertainty.
          
    
    Members:
    
      L2Loss
    
      IntensityWeightedLoss
    
      L1Loss
    
      LogL2Loss
    
      WeightedL2Loss
    """
    L1Loss: typing.ClassVar[LossType]  # value = <LossType.L1Loss: 2>
    L2Loss: typing.ClassVar[LossType]  # value = <LossType.L2Loss: 0>
    LogL2Loss: typing.ClassVar[LossType]  # value = <LossType.LogL2Loss: 3>
    IntensityWeightedLoss: typing.ClassVar[LossType]  # value = <LossType.IntensityWeightedLoss: 1>
    WeightedL2Loss: typing.ClassVar[LossType]  # value = <LossType.WeightedL2Loss: 4>
    __members__: typing.ClassVar[dict[str, LossType]]  # value = {'L2Loss': <LossType.L2Loss: 0>, 'IntensityWeightedLoss': <LossType.IntensityWeightedLoss: 1>, 'L1Loss': <LossType.L1Loss: 2>, 'LogL2Loss': <LossType.LogL2Loss: 3>, 'WeightedL2Loss': <LossType.WeightedL2Loss: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NormalizationType:
    """
    
              Enumeration of normalization strategies for spectral fitting.
    
              Controls how simulated and experimental spectra are brought to a
              common scale before computing the loss function.
    
              Attributes
              ----------
              MaxNorm
                  Legacy normalization: independently divide each spectrum by its
                  maximum value. Simple but can distort relative peak heights.
              AnalyticNorm
                  Analytically optimal scale factor. Computes c = (y_sim · y_exp) /
                  (y_exp · y_exp) that minimizes ||y_sim - c * y_exp||². Fully
                  differentiable and better for gradient-based optimization.
              OptimizedScaleNorm
                  Free scale factor optimized via gradient descent. Same structure
                  as AnalyticNorm but `c` is a free parameter rather than
                  analytically computed. Initialize via Optimizer which sets
                  scale_factor from a first analytic estimate.
          
    
    Members:
    
      MaxNorm
    
      AnalyticNorm
    
      OptimizedScaleNorm
    """
    AnalyticNorm: typing.ClassVar[NormalizationType]  # value = <NormalizationType.AnalyticNorm: 1>
    MaxNorm: typing.ClassVar[NormalizationType]  # value = <NormalizationType.MaxNorm: 0>
    OptimizedScaleNorm: typing.ClassVar[NormalizationType]  # value = <NormalizationType.OptimizedScaleNorm: 2>
    __members__: typing.ClassVar[dict[str, NormalizationType]]  # value = {'MaxNorm': <NormalizationType.MaxNorm: 0>, 'AnalyticNorm': <NormalizationType.AnalyticNorm: 1>, 'OptimizedScaleNorm': <NormalizationType.OptimizedScaleNorm: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PlasmaModel:
    """
    
              Abstract base class for plasma models.
    
              This class defines the interface for all plasma models in LIBSkit.
              It provides methods for calculating plasma properties and spectral
              characteristics.
          
    """
    def build_fine_wavelength_grid(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
                       Build a uniform fine wavelength grid for accurate spectral computation.
        
                       Constructs a grid spanning the same range as the input wavelengths,
                       with spacing determined by the narrowest emission line's FWHM.
                       If the input grid is already fine enough, the spacing matches
                       the input.
        
                       Parameters
                       ----------
                       wavelengths : numpy.ndarray
                           Wavelength range defining the desired extent.
        
                       Returns
                       -------
                       numpy.ndarray
                           Fine uniform wavelength grid.
        
                       Example
                       -------
                       >>> wl_fine = model.build_fine_wavelength_grid(wavelengths)
                       >>> irr_fine = model.calculate_irradiance(wl_fine)
        """
    def get_fine_grid_spacing(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> float:
        """
                       Get the fine grid spacing determined by the narrowest emission line.
        
                       Returns the wavelength step size (in nm) needed to adequately
                       sample all emission lines: min_voigt_fwhm / fine_samples_per_fwhm.
        
                       Parameters
                       ----------
                       wavelengths : numpy.ndarray
                           Wavelength range to consider for determining the narrowest line.
        
                       Returns
                       -------
                       float
                           Fine grid spacing in nm.
        
                       Example
                       -------
                       >>> dl = model.get_fine_grid_spacing(wavelengths)
                       >>> print(f"Fine grid spacing: {dl:.5f} nm")
        """
    def get_minimum_voigt_fwhm(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> float:
        """
                       Get the minimum Voigt FWHM across all emission lines in the range.
        
                       Parameters
                       ----------
                       wavelengths : numpy.ndarray
                           Wavelength range to consider.
        
                       Returns
                       -------
                       float
                           Minimum Voigt FWHM in nm.
        """
class SlopingSpectrometer(Instrument):
    """
    
              Spectrometer with wavelength-dependent Gaussian broadening.
    
              Models instrumental broadening as a Gaussian whose FWHM grows
              linearly with wavelength: FWHM(λ) = m·λ + b.
    
              Uses a change-of-variable trick to convert the position-dependent
              convolution into a shift-invariant one, enabling efficient FFT-based
              computation in O(N log N) instead of O(N²).
    
              Parameters
              ----------
              fwhm_slope : float
                  Slope m of FWHM(λ) = m·λ + b  (nm/nm, dimensionless).
              fwhm_intercept : float
                  Intercept b of FWHM(λ) = m·λ + b  (nm).
    
              Attributes
              ----------
              fwhm_slope : float
                  Slope of the linear FWHM model.
              fwhm_intercept : float
                  Intercept of the linear FWHM model.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>> spec = libskit.SlopingSpectrometer(fwhm_slope=1.5e-4, fwhm_intercept=7.8e-3)
              >>> model = libskit.LTEModel({"Fe": 1.0}, 10000.0, 1e23, 1e-3)
              >>> wavelengths = np.linspace(200, 900, 5000)
              >>> spectrum = spec.calculate_spectrum(wavelengths, model)
          
    """
    def __init__(self, fwhm_slope: typing.SupportsFloat | typing.SupportsIndex, fwhm_intercept: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def calculate_spectrum(self, wavelengths: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], model: PlasmaModel) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    @property
    def fwhm_intercept(self) -> float:
        ...
    @fwhm_intercept.setter
    def fwhm_intercept(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fwhm_slope(self) -> float:
        ...
    @fwhm_slope.setter
    def fwhm_slope(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Spectrum:
    """
    
              Container for spectroscopic data with wavelength-intensity pairs.
    
              A Spectrum object stores matched arrays of wavelength and intensity
              values, representing either measured or simulated spectroscopic data.
    
              Parameters
              ----------
              wavelengths : numpy.ndarray
                  1D array of wavelength values in nanometers. Must be the same
                  length as irradiance.
              irradiance : numpy.ndarray
                  1D array of intensity/irradiance values corresponding to each
                  wavelength point.
    
              Attributes
              ----------
              wavelengths : numpy.ndarray
                  Wavelength array in nanometers.
              irradiance : numpy.ndarray
                  Intensity/irradiance array.
    
              Example
              -------
              >>> import libskit
              >>> import numpy as np
              >>> wl = np.linspace(200, 900, 1000)
              >>> intensity = np.random.rand(1000)
              >>> spectrum = libskit.Spectrum(wl, intensity)
          
    """
    def __init__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    @property
    def irradiance(self) -> typing.Any:
        ...
    @irradiance.setter
    def irradiance(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    @property
    def wavelengths(self) -> typing.Any:
        ...
    @wavelengths.setter
    def wavelengths(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
def blackbody_irradiance(arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
               Spectral radiance of a blackbody at temperature ``T``.
    
               Evaluates Planck's law for the supplied wavelength grid,
               returning the per-wavelength spectral radiance in SI units.
    
               Args:
                   wavelengths: Wavelengths in nm (numpy array).
                   T: Blackbody temperature in K.
    
               Returns:
                   numpy.ndarray: Spectral radiance B\\u03bb(T) in W m\\u207b\\u00b3 sr\\u207b\\u00b9.
    """
def calculate_irradiance_test(arg0: PlasmaModel, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
               Reference irradiance calculation used by the test suite.
    
               Evaluates the analytical Planck blackbody irradiance plus a
               narrow Gaussian emission line on a fixed wavelength grid; the
               Python tests compare the result against ``blackbody_irradiance``
               plus a numerical convolution to detect regressions in the
               wavelength-grid handling.
    
               Returns:
                   numpy.ndarray: Irradiance in W m\\u207b\\u00b2 nm\\u207b\\u00b9 sr\\u207b\\u00b9.
    """
def calculate_partition_function(arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
               Calculate the partition function for a given ion and temperature.
    
               Uses the full energy level data from the NIST database.
               U(T) = Σ g_i * exp(-E_i / (k_B * T))
    
               Args:
                   ion (str): The ion species, e.g. "Ca I", "Fe II"
                   temperature (float): Temperature in Kelvin
    
               Returns:
                   float: The partition function value
    
               Example:
                   >>> libskit.calculate_partition_function("Ca I", 10000.0)
    """
def calculate_saha_function(arg0: str, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
               Calculate the Saha function for a given element and ionization level.
    
               Args:
                   element (str): Element symbol, e.g. "Ca", "Fe"
                   ionization_level (int): Ionization level (0 = neutral to singly ionized)
                   temperature (float): Temperature in Kelvin
    
               Returns:
                   float: The Saha function value in m^-3
    
               Example:
                   >>> libskit.calculate_saha_function("Ca", 0, 10000.0)
    """
def compute_loss_enzyme(arg0: Experiment) -> float:
    """
               Compute loss using the Enzyme numerical pipeline.
    
               Useful for verifying that the reimplemented pipeline
               matches the standard loss() function.
    
               Args:
                   experiment: An Experiment object (double precision).
    
               Returns:
                   float: The computed loss value.
    """
def get_accuracy() -> float:
    """
               Get the current accuracy threshold.
    
               Returns:
                   float: Fraction of max peak height used as noise floor
    """
def get_aki_accuracies() -> dict[str, list[float]]:
    """
               Get NIST accuracy fractions for all Einstein A coefficients.
    
               Returns a dict mapping ion names to lists of accuracy fractions,
               one per emission line in the database (in the same order expected
               by ``set_aki_multipliers``).
    
               Returns:
                   dict[str, list[float]]: {ion: [acc_fraction, ...]}
    """
def get_database_path() -> str:
    """
               Get the current path to the atomic databases directory.
    
               Returns:
                   str: Current database path
    """
def get_fine_samples_per_fwhm() -> float:
    """
               Get the current number of fine grid samples per narrowest line FWHM.
    
               Returns:
                   float: Current fine grid samples per FWHM
    """
def get_gradients_enzyme(arg0: Experiment) -> dict[str, float]:
    """
               Calculate gradients using Enzyme reverse-mode AD (recommended default).
    
               Computes all gradients of the loss function w.r.t. plasma model
               parameters in a single reverse-mode pass.  Cost is O(1) in the
               number of parameters, making it significantly faster than the
               forward-mode get_gradients_forward() for models with many parameters.
    
               Args:
                   experiment: An Experiment object (double precision).
    
               Returns:
                   dict: Mapping of parameter names to gradient values.
    """
def get_gradients_forward(arg0: Experiment) -> dict[str, float]:
    """
               Calculate gradients using forward-mode AD (reference implementation).
    
               Computes gradients of the loss function w.r.t. all plasma model
               parameters using forward-mode automatic differentiation.  Runtime
               is O(N) in the number of parameters.
    
               Prefer get_gradients_enzyme() for production use — it computes the
               same result in a single reverse-mode pass regardless of parameter
               count.
    
               Args:
                   experiment: An Experiment object (double precision).
    
               Returns:
                   dict: Mapping of parameter names to gradient values.
    """
def load_databases(verbose: typing.SupportsInt | typing.SupportsIndex = 1) -> None:
    """
               Manually trigger loading of atomic databases.
    
               This is usually called automatically when needed, but can be called
               explicitly to preload databases or check for loading errors.
    
               Parameters:
                   verbose: Verbosity level (0=silent, 1=normal). Default: 1
    """
def loss(arg0: Experiment) -> float:
    """
               Calculate the loss function value for an experiment.
    
               Takes an Experiment object and returns the scalar loss value
               computed using the experiment's specified loss function type
               (L2Loss or IntensityWeightedLoss).
    
               Args:
                   experiment: An Experiment object containing the plasma model,
                              instrument, target spectrum, and loss type.
    
               Returns:
                   float: The computed loss value.
    
               Example:
                   >>> loss_value = libskit.loss(experiment)
    """
def optimal_scale_factor(arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> float:
    """
               Compute the analytically optimal scale factor between two spectra.
    
               Finds c that minimizes ||y_sim - c * y_ref||²:
                 c_opt = (y_sim · y_ref) / (y_ref · y_ref)
    
               Args:
                   y_sim: Simulated spectrum intensities (numpy array).
                   y_ref: Reference (experimental) spectrum intensities (numpy array).
    
               Returns:
                   float: Optimal scale factor c.
    """
def reload_databases() -> None:
    """
               Force reload of atomic databases from current database path.
    
               Useful after changing the database path with set_database_path().
    """
def reset_aki() -> None:
    """
               Restore all Einstein A coefficients to their original database values.
    
               Undoes any modifications made by ``set_aki_multipliers()``.
    """
def set_accuracy(arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
               Set the accuracy threshold as a fraction of the maximum peak height.
    
               This single parameter controls both:
               - Which emission lines are included (lines whose peak height is
                 below accuracy * max_peak_height are discarded)
               - Where broadening wings are truncated (broadening stops when the
                 spectral density drops below accuracy * max_peak_height)
    
               Args:
                   accuracy (float): Fraction of max peak height (default: 1e-5)
    
               Example:
                   libskit.set_accuracy(1e-3)  # Noise floor at 0.1% of tallest peak
    """
def set_aki_multipliers(arg0: collections.abc.Mapping[str, collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]]) -> None:
    """
               Multiply each emission line's Einstein A coefficient by a factor.
    
               For each ion key in the dictionary, the corresponding list of
               factors is applied element-wise to the lines of that ion.  The
               factor list must have the same length as the number of lines
               for that ion.
    
               Use ``reset_aki()`` to restore the original values.
    
               Args:
                   multipliers (dict[str, list[float]]): Mapping from ion name
                       (e.g. ``"Ca II"``) to a list of multiplicative factors.
    
               Raises:
                   RuntimeError: If an ion name is not found or the factor list
                       has the wrong length.
    
               Example:
                   >>> factors = {"Ca II": [1.05, 0.97, 1.02, ...]}
                   >>> libskit.set_aki_multipliers(factors)
    """
def set_database_path(arg0: str) -> None:
    """
               Set the path to the atomic databases directory.
    
               Args:
                   path (str): Path to directory containing the CSV database files
                              (lines_processed_all.csv, ionization_energies_processed.csv,
                               levels_processed.csv)
    
               Example:
                   libskit.set_database_path("/path/to/my/databases")
    """
def set_fine_samples_per_fwhm(arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
               Set the number of fine grid samples per narrowest line FWHM.
    
               Controls the resolution of the internal fine grid used for
               FFT-based instrumental broadening. Higher values give more
               accurate convolution at the cost of computation time.
    
               Args:
                   samples (float): Number of samples per FWHM (default: 5). Values < 1 undersample the narrowest line.
    
               Example:
                   libskit.set_fine_samples_per_fwhm(10)  # Higher accuracy
    """
AnalyticNorm: NormalizationType  # value = <NormalizationType.AnalyticNorm: 1>
ENZYME_ENABLED: bool = True
L1Loss: LossType  # value = <LossType.L1Loss: 2>
L2Loss: LossType  # value = <LossType.L2Loss: 0>
LogL2Loss: LossType  # value = <LossType.LogL2Loss: 3>
MaxNorm: NormalizationType  # value = <NormalizationType.MaxNorm: 0>
OptimizedScaleNorm: NormalizationType  # value = <NormalizationType.OptimizedScaleNorm: 2>
IntensityWeightedLoss: LossType  # value = <LossType.IntensityWeightedLoss: 1>
WeightedL2Loss: LossType  # value = <LossType.WeightedL2Loss: 4>
__version__: str = 'dev'
