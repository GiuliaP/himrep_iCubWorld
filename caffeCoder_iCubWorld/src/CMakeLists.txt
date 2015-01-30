# Copyright: (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.8)

set(PROJECTNAME caffeCoder)
project(${PROJECTNAME})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

option(CPU_ONLY "Use CPU only for Caffe" OFF)
option(USE_CUDNN "Use CuDNN library for Caffe" ON)

find_package(OpenCV REQUIRED)

find_package(Caffe REQUIRED)

if(NOT CPU_ONLY)
  find_package(CUDA)

  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
endif()

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${boostMIL_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${Caffe_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

# no need to use the following instruction because this is automatically done by CUDA_ADD_EXECUTABLE
if(NOT CPU_ONLY)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

set(folder_source main.cpp)
set(folder_header CaffeFeatExtractor.hpp)
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})


if(NOT CPU_ONLY)
  cuda_add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
else()
  add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
endif()

target_link_libraries(${PROJECTNAME} ${YARP_LIBRARIES} boostMIL ${OpenCV_LIBRARIES})

target_link_libraries(${PROJECTNAME} ${Caffe_LIBRARIES})

if(NOT CPU_ONLY)
target_link_libraries(${PROJECTNAME} ${CUDA_LIBRARIES})
endif()

install(TARGETS ${PROJECTNAME} DESTINATION bin)

