Tutorials
=========

LIBSkit includes a series of interactive Jupyter notebooks that walk through
the library's features step by step.  The notebooks live in the ``notebooks/``
directory of the repository.

Deeper scientific exploration notebooks (convergence studies, uncertainty
propagation, ChemCam reanalysis, multiplet Boltzmann methods, autodiff
comparisons, etc.) live in the companion `libskit_experiments`_ repository
organised by topic.

.. _libskit_experiments: https://codebase.helmholtz.cloud/christoph.egerland/libskit-experiments

Tutorial 1 – Database Loading and Verification
-----------------------------------------------

Learn how LIBSkit manages the atomic databases that underpin every
calculation.  Topics include inspecting the bundled database path,
verifying file integrity, and overriding the database path when
needed.

.. code-block:: python

    import libskit

    # Databases are loaded automatically on import.
    print(libskit.get_database_path())

    # Verify database integrity
    from libskit.database_verification import verify_databases
    result = verify_databases(libskit.get_database_path())
    print(result.is_valid)

**Key APIs:** ``get_database_path``, ``set_database_path``, ``load_databases``,
``database_verification.verify_databases``

Tutorial 2 – LTE Spectrum Modelling
------------------------------------

Create Local Thermodynamic Equilibrium (LTE) plasma models and
compute synthetic LIBS spectra.  Explore how temperature, electron
density, and elemental composition influence the spectral output.

.. code-block:: python

    import libskit
    import numpy as np

    model = libskit.LTEModel(
        {"Si": 0.72, "Na": 0.09, "Ca": 0.19},
        17300.0, 2.53e23, 2.4e-3
    )

    wavelengths = np.arange(250, 900, 0.05)
    instrument  = libskit.GenericSpectrometer(0.05, 0.001)
    spectrum    = instrument.calculate_spectrum(wavelengths, model)

**Key APIs:** ``LTEModel``, ``GenericSpectrometer``, ``calculate_irradiance``,
``get_population_fractions``

Tutorial 4 – Parameter Optimisation
------------------------------------

Fit LTE model parameters to a measured spectrum using gradient-based
(L-BFGS) and gradient-free (Nelder-Mead) optimisers.  Includes
convergence analysis with multi-start Latin Hypercube Sampling.

.. code-block:: python

    import libskit

    experiment = libskit.Experiment(
        libskit.LossType.IntensityWeightedLoss,
        initial_model, instrument, target_spectrum
    )

    opt = libskit.Optimizer(experiment, algorithm="LD_LBFGS")
    result = opt.optimize()

**Key APIs:** ``Experiment``, ``Optimizer``, ``LossType``,
``convergence_analysis``, ``visualization.plot_optimization_results``

Tutorial 5 – Multi-Layer Plasma Modelling
-----------------------------------------

Build realistic multi-layer plasma simulations where temperature,
density, and composition vary along the optical path.

.. code-block:: python

    import libskit

    layer1 = libskit.LTEModel({"Fe": 1.0}, 12000, 1e23, 1e-3)
    layer2 = libskit.LTEModel({"Fe": 0.8, "Ca": 0.2}, 9000, 5e22, 2e-3)
    layers = libskit.LTELayers([layer1, layer2])

    spectrum = instrument.calculate_spectrum(wavelengths, layers)

**Key APIs:** ``LTELayers``, ``get_parameter_names``, ``get_parameter``,
``set_parameter``

Tutorial 6 – Time-Integrated Modelling
---------------------------------------

Simulate time-integrated LIBS spectra where plasma temperature and
electron density decay asymptotically during the detector gate.

.. code-block:: python

    import libskit

    ti_model = libskit.LTETimeIntegrated(
        {"Fe": 1.0},
        start_temperature=13000, end_temperature=6500,
        tau_temperature=195e-9,
        start_electron_density=4e23, end_electron_density=0.8e23,
        tau_electron_density=90e-9,
        start_length=0.5e-3, end_length=3.0e-3,
        tau_length=109e-9,
        integration_time=1990e-9
    )

    spectrum = instrument.calculate_spectrum(wavelengths, ti_model)

**Key APIs:** ``LTETimeIntegrated``

Tutorial 7 – Loading Measurement Data
--------------------------------------

Load experimental LIBS spectra from Aryelle echelle spectrometers,
apply wavelength calibration and order stitching, and compare with
LTE simulations.

.. code-block:: python

    import libskit

    reader  = libskit.AryReader("measurement.ary")
    spec    = reader.spectrum()
    cropped = spec.crop(350, 450).normalize("max").to_vacuum()

**Key APIs:** ``AryReader``, ``Spectrum``, ``SlopingSpectrometer``,
``vacuum_to_air``, ``air_to_vacuum``
