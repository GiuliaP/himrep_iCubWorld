# Copyright: 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.6)

project(himrep_iCubWorld)

find_package(YARP REQUIRED)
find_package(ICUB REQUIRED)
find_package(ICUBcontrib REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH} ${ICUB_MODULE_PATH} ${ICUBCONTRIB_MODULE_PATH})

include(YarpInstallationHelpers)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()

add_subdirectory(linearClassifierModule_iCubWorld)
add_subdirectory(caffeCoder_iCubWorld)
#add_subdirectory(sparseCoder_iCubWorld)

icubcontrib_add_uninstall_target()

