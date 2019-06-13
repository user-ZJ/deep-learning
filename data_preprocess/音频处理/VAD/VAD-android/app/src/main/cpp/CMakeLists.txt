# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        libfvad

        # Sets the library as a shared library.
        SHARED
        signal_processing/division_operations.c
        signal_processing/energy.c
        signal_processing/get_scaling_square.c
        signal_processing/resample_48khz.c
        signal_processing/resample_by_2_internal.c
        signal_processing/resample_by_2_internal.h
        signal_processing/resample_fractional.c
        signal_processing/signal_processing_library.h
        signal_processing/spl_inl.c
        signal_processing/spl_inl.h
        vad/vad_core.c
        vad/vad_core.h
        vad/vad_filterbank.c
        vad/vad_filterbank.h
        vad/vad_gmm.c
        vad/vad_gmm.h
        vad/vad_sp.c
        vad/vad_sp.h
        common.h
        fvad.c
        fvad.h
        # Provides a relative path to your source file(s).
        vad-lib.cpp)




# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        libfvad

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})