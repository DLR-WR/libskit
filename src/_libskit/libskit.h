/*
 * Copyright (c) 2025 Christoph H. Egerland
 * SPDX-License-Identifier: MIT
 */

/**
 * @file libskit.h
 * @brief Main header file for the libskit spectroscopy library.
 *
 * This header provides access to all major components of libskit.
 *
 */
#ifndef libskit_H
#define libskit_H

#include "AtomicData/atom_info.h"
#include "AtomicData/constants.h"
#include "AtomicData/elemental_data.h"
#include "PlasmaModel/LTEModel.h"
#include "PlasmaModel/LTELayers.h"
#include "PlasmaModel/LTESharedLayers.h"
#include "PlasmaModel/LTETimeIntegrated.h"
#include "PlasmaModel/LTETwoTime.h"
#include "PlasmaModel/PlasmaModel.h"
#include "Instrument/Instrument.h"
#include "Instrument/GenericSpectrometer.h"
#include "Instrument/SlopingSpectrometer.h"
#include "Instrument/Aryelle.h"
#include "Instrument/ChemCam.h"
#include "Spectrum/Spectrum.h"
#include "functions.h"
#include "gradients.h"
// TODO: Faddeeva.hh is no longer used in the main code (replaced by McLean
// Voigt approximation in voigt_profile_approx). Consider removing Faddeeva.hh,
// Faddeeva.cc, and the faddeeva_test target.
#include "utils/Faddeeva.hh"
#include "utils/arange.h"
#include "utils/to_roman.h"

#endif /* libskit_H */
