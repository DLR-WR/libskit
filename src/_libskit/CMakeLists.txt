set(libskit_sources
    gradients.cpp
    AtomicData/elemental_data.cpp
    Instrument/Aryelle.cpp
    Instrument/ChemCam.cpp
    Instrument/GenericSpectrometer.cpp
    Instrument/SlopingSpectrometer.cpp
    Instrument/FFT.cpp
    PlasmaModel/LTEModel.cpp
    PlasmaModel/LTELayers.cpp
    PlasmaModel/LTESharedLayers.cpp
    PlasmaModel/LTETimeIntegrated.cpp
    PlasmaModel/LTETwoTime.cpp
    utils/arange.cpp
    utils/Faddeeva.cc
    utils/to_roman.cpp
    )

add_library(libskit STATIC ${libskit_sources})
target_link_libraries(libskit Eigen3::Eigen)
set_target_properties(libskit PROPERTIES PUBLIC_HEADER libskit.h POSITION_INDEPENDENT_CODE ON)

# ── Enzyme reverse-mode AD support (required) ────────────────────────
target_sources(libskit PRIVATE enzyme_gradients.cpp)
set_source_files_properties(enzyme_gradients.cpp PROPERTIES
    COMPILE_FLAGS "${ENZYME_PLUGIN_COMPILE_FLAGS}"
)
target_compile_definitions(libskit PUBLIC ENZYME_ENABLED)
