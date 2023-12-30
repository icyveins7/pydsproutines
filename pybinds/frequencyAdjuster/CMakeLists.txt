cmake_minimum_required(VERSION 3.20)

project(pydsp_frequencyAdjuster)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64") # Need this to work with intel stuff for m1/m2
    message("CMAKE_OSX_ARCHITECTURES is " ${CMAKE_OSX_ARCHITECTURES})
endif()

include_directories($ENV{INCLUDE} $ENV{CPATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../ipp_ext/include)

# Define the library paths for IPP
if (WIN32)
    message("Finding IPP libraries for Windows...")
    find_library(ippcorelib NAMES ippcore ENV LIBRARY_PATH)
    find_library(ippslib NAMES ipps ENV LIBRARY_PATH) # Did i actually need HINTS here too?
    find_library(ippvmlib NAMES ippvm ENV LIBRARY_PATH)
    message("Libpath: " ${ippcorelib})
    message("Libpath: " ${ippslib})
    message("Libpath: " ${ippvmlib})
elseif (APPLE)
    # For Apple Silicon builds, the linker seems to dislike
    # dynamic linking, but static linking seems to work
    message("Finding IPP libraries for MacOS...")
    # Make sure you leave the .a in the hint, otherwise it reverts to using dylib by default
    find_library(ippcorelib NAMES libippcore.a HINTS ENV DYLD_LIBRARY_PATH) # on mac it seems like it's here, even though it's linux based
    find_library(ippslib NAMES libipps.a HINTS ENV DYLD_LIBRARY_PATH)
    find_library(ippvmlib NAMES libippvm.a HINTS ENV DYLD_LIBRARY_PATH)
    message("Libpath: " ${ippcorelib})
    message("Libpath: " ${ippslib})
    message("Libpath: " ${ippvmlib})
    # I don't need rpath any more because it's static?
    # I will leave this below as a reminder that you don't need to use RPATH
    # For mac we must also find the library directory and set rpath. The linux method doesn't seem to compile properly
    # get_filename_component(ipplibdir ${ippcorelib} DIRECTORY)
    # set(CMAKE_BUILD_RPATH ${ipplibdir})
    # message("Using rpath: " ${CMAKE_BUILD_RPATH})
    # set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON")
else ()
    message("Finding IPP libraries for Linux...")
    find_library(ippcorelib NAMES libippcore ippcore HINTS ENV LD_LIBRARY_PATH)
    find_library(ippslib NAMES libipps ipps HINTS ENV LD_LIBRARY_PATH)
    find_library(ippvmlib NAMES libvmlib ipps HINTS ENV LD_LIBRARY_PATH)
    message("Libpath: " ${ippcorelib})
    message("Libpath: " ${ippslib})
    message("Libpath: " ${ippvmlib})
endif()

# Define test executable for vectors
add_executable(test_frequencyAdjuster frequencyAdjuster.cpp)
target_link_libraries(test_frequencyAdjuster PUBLIC ${ippcorelib} ${ippslib} ${ippvmlib})


