API Documentation
=================

This section provides comprehensive API documentation for LIBSkit.

For Python API documentation, see :doc:`python_api`.


C++ API Reference
-----------------

The C++ library provides the core computational engine for spectrum simulation.
The C++ code is organized into several modules:

Core Functions
~~~~~~~~~~~~~~

Physical calculations and atomic physics utilities.

.. doxygenfunction:: libskit::calculate_debroglie_wavelength
   :project: libskit

.. doxygenfunction:: libskit::calculate_partition_function
   :project: libskit

.. doxygenfunction:: libskit::calculate_saha_function
   :project: libskit

.. doxygenfunction:: libskit::calculate_population_levels
   :project: libskit

.. doxygenfunction:: libskit::blackbody_radiation
   :project: libskit

Line Profile Functions
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: libskit::voigt_profile
   :project: libskit

.. doxygenfunction:: libskit::voigt_profile_approx
   :project: libskit

.. doxygenfunction:: libskit::get_doppler_standard_deviation
   :project: libskit

.. doxygenfunction:: libskit::get_stark_broadening_fwhm
   :project: libskit

.. doxygenfunction:: libskit::get_stark_shift
   :project: libskit


Plasma Models
~~~~~~~~~~~~~

Abstract base class and implementations for plasma emission models.

PlasmaModel
^^^^^^^^^^^

.. doxygenstruct:: libskit::PlasmaModel
   :project: libskit
   :members:

LTEModel
^^^^^^^^

.. doxygenstruct:: libskit::LTEModel
   :project: libskit
   :members:

LTELayers
^^^^^^^^^

.. doxygenstruct:: libskit::LTELayers
   :project: libskit
   :members:

LTETimeIntegrated
^^^^^^^^^^^^^^^^^

.. doxygenstruct:: libskit::LTETimeIntegrated
   :project: libskit
   :members:


Instrument Models
~~~~~~~~~~~~~~~~~

Abstract base class and implementations for spectrometer models.

Instrument
^^^^^^^^^^

.. doxygenclass:: libskit::Instrument
   :project: libskit
   :members:

GenericSpectrometer
^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: libskit::GenericSpectrometer
   :project: libskit
   :members:


Data Structures
~~~~~~~~~~~~~~~

Spectrum
^^^^^^^^

.. doxygenstruct:: libskit::Spectrum
   :project: libskit
   :members:

Experiment
^^^^^^^^^^

.. doxygenstruct:: libskit::Experiment
   :project: libskit
   :members:

Enumerations
~~~~~~~~~~~~

.. doxygenenum:: libskit::LossType
   :project: libskit


Building the C++ Documentation
------------------------------

To regenerate the C++ API documentation:

.. code-block:: bash

   cd docs/source
   doxygen Doxyfile
   cd ..
   make html

This will:

1. Run Doxygen to generate XML documentation from the C++ source
2. Use Breathe to integrate the Doxygen output into Sphinx
3. Build the complete HTML documentation
