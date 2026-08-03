# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# Find Google Test
find_package(GTest REQUIRED)

# Add the comprehensive LTEModel test executable
add_executable(test_lte_model test_lte_model.cpp)
target_link_libraries(test_lte_model PRIVATE GTest::GTest GTest::Main libskit)

# Add comprehensive Spectrum tests
add_executable(test_spectrum test_spectrum.cpp)
target_link_libraries(test_spectrum PRIVATE GTest::GTest GTest::Main libskit)

# Add comprehensive GenericSpectrometer tests
add_executable(test_generic_spectrometer test_generic_spectrometer.cpp)
target_link_libraries(test_generic_spectrometer PRIVATE GTest::GTest GTest::Main libskit)

# Add comprehensive utility function tests
add_executable(test_utility_functions test_utility_functions.cpp)
target_link_libraries(test_utility_functions PRIVATE GTest::GTest GTest::Main libskit)

# Add LTELayers equivalence tests
add_executable(test_lte_layers test_lte_layers.cpp)
target_link_libraries(test_lte_layers PRIVATE GTest::GTest GTest::Main libskit)

# Add LTESharedLayers tests
add_executable(test_lte_shared_layers test_lte_shared_layers.cpp)
target_link_libraries(test_lte_shared_layers PRIVATE GTest::GTest GTest::Main libskit)

# Add FFT primitives tests
add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft PRIVATE GTest::GTest GTest::Main libskit Eigen3::Eigen)

# Add exception-path tests for PlasmaModel implementations
add_executable(test_error_paths test_error_paths.cpp)
target_link_libraries(test_error_paths PRIVATE GTest::GTest GTest::Main libskit)

# Add test executables to the Google Test test suite
enable_testing()
add_test(NAME test_lte_model COMMAND test_lte_model WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_spectrum COMMAND test_spectrum WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_generic_spectrometer COMMAND test_generic_spectrometer WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_utility_functions COMMAND test_utility_functions WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_fft COMMAND test_fft WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_error_paths COMMAND test_error_paths WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_lte_shared_layers COMMAND test_lte_shared_layers WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add Python tests (requires Python and the built module)
find_package(Python COMPONENTS Interpreter)
if(Python_FOUND)
    # Test the comprehensive LTEModel Python tests
    add_test(
        NAME test_lte_model_python
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_lte_model_python.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # Test the comprehensive Python integration tests
    add_test(
        NAME test_comprehensive_python
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_comprehensive_python.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # Set environment variables for Python tests
    set_tests_properties(test_lte_model_python PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
    set_tests_properties(test_comprehensive_python PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
endif()

