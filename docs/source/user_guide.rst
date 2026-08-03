User Guide
==========

This guide covers the main concepts and usage patterns of LIBSkit.

Overview
--------

LIBSkit is designed for analyzing Laser-Induced Breakdown Spectroscopy (LIBS) data.
It provides tools for:

* Simulating LIBS spectra from known compositions
* Fitting plasma parameters via gradient-based optimisation
* Processing measurement data
Core Components
---------------


LTE Model
~~~~~~~~~

The Local Thermodynamic Equilibrium model handles plasma physics calculations in LTE conditions:

.. code-block:: python

    import libskit

    model = libskit.LTEModel(
        {"Si": 0.72, "Na": 0.09, "Ca": 0.19},  # element mass ratios
        17300.0,   # temperature in K
        2.53e23,   # electron density in m^-3
        2.4e-3     # plasma radius / length in m
    )

For advanced scenarios, LIBSkit provides two additional plasma models:

* **LTELayers** — stacks multiple ``LTEModel`` instances along the optical
  path, solving the radiative transfer equation layer by layer.
* **LTETimeIntegrated** — integrates over the detector gate with
  asymptotic decay of temperature and electron density, and saturating
  expansion of plasma length.


Spectrometer Modelling
~~~~~~~~~~~~~~~~~~~~~~

The ``GenericSpectrometer`` class models instrument response by convolving the irradiance from the plasma with a Voigt profile:

.. code-block:: python

    instrument = libskit.GenericSpectrometer(0.05, 0.001)

    wavelengths = np.arange(250, 900, 0.05)
    spectrum = instrument.calculate_spectrum(wavelengths, model)

For instruments with wavelength-dependent broadening, use ``SlopingSpectrometer``
with callable sigma/gamma functions.

Instrumental Broadening
^^^^^^^^^^^^^^^^^^^^^^^

For specific instruments, you may need to implement custom broadening functions. Here's an example for the Butterfly spectrometer:

.. code-block:: python

    def instrumental_broadening_fcn_butterfly_sigma(self, w):
        """Gaussian component (sigma) for Butterfly spectrometer."""
        instrumental_broadening_pars = [2.67710854e-05, 3.24293535e-03]
        return (instrumental_broadening_pars[0] * w + instrumental_broadening_pars[1])

    def instrumental_broadening_fcn_butterfly_gamma(self, w):
        """Lorentzian component (gamma) for Butterfly spectrometer."""
        instrumental_broadening_pars = [2.54260506e-05, 4.77688359e-03]
        return (instrumental_broadening_pars[0] * w + instrumental_broadening_pars[1])

.. note::
   Be aware that the first function returns sigma for the Gaussian part,
   while the second returns FWHM for the Lorentzian part of the Voigt profile.

Atomic Data
-----------

LIBSkit bundles three NIST ASD databases that are installed alongside the
package and loaded automatically on ``import libskit``:

* **lines_processed_all.csv** — emission line wavelengths, Einstein A
  coefficients, energy levels, statistical weights, and Stark broadening
  parameters from Stark-B.
* **ionization_energies_processed.csv** — ionization potentials for all
  species.
* **levels_processed.csv** — energy levels and statistical weights used
  for partition function calculations.

The bundled path can be inspected at any time:

.. code-block:: python

    print(libskit.get_database_path())

To use a custom database directory instead, call ``set_database_path()``
followed by ``reload_databases()``.

For Stark broadening parameters the Stark-B database is used as well as
some manually added parameters.

Optimisation
------------

The ``Optimizer`` class wraps NLopt algorithms for fitting model parameters
to experimental spectra.  Both gradient-based (L-BFGS) and gradient-free
(Nelder-Mead) strategies are supported:

.. code-block:: python

    experiment = libskit.Experiment(
        libskit.LossType.IntensityWeightedLoss,
        initial_model, instrument, target_spectrum
    )

    opt = libskit.Optimizer(experiment, algorithm="LD_LBFGS")
    result = opt.optimize()

A standalone ``custom_gradient_descent()`` function is also available for
fine-grained control over the optimisation loop.

Spectral Analysis
-----------------

Peak detection, Voigt fitting, and noise estimation utilities live in
``libskit.spectral_analysis``:

.. code-block:: python

    from libskit.spectral_analysis import find_spectrum_peaks, fit_voigt_profile

    peaks = find_spectrum_peaks(wavelengths, spectrum)
    fit   = fit_voigt_profile(wavelengths, spectrum, line_center=400.0)
