Quick Start
===========

This guide will get you up and running with LIBSkit in just a few minutes.

Basic Usage
-----------

Databases are loaded automatically on ``import libskit`` — no setup needed.

.. code-block:: python

    import libskit
    import numpy as np

    # Create an LTE plasma model
    model = libskit.LTEModel(
        {"Si": 0.72, "Na": 0.09, "Ca": 0.19},  # element mass ratios
        17300.0,   # temperature in K
        2.53e23,   # electron density in m^-3
        2.4e-3     # plasma radius/length in m
    )

    # Define an instrument (Voigt broadening)
    instrument = libskit.GenericSpectrometer(
        0.05,   # Gaussian sigma in nm
        0.001   # Lorentzian gamma in nm
    )

    # Calculate a spectrum
    wavelengths = np.arange(250, 900, 0.05)  # nm
    spectrum = instrument.calculate_spectrum(wavelengths, model)

    # Plot
    import matplotlib.pyplot as plt
    plt.plot(spectrum.wavelengths, spectrum.irradiance)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Irradiance")
    plt.show()


Next Steps
----------

* Read the :doc:`user_guide` for detailed explanations of each component
* Work through the :doc:`tutorials` for step-by-step examples
* Check the :doc:`python_api` for the complete Python API reference
