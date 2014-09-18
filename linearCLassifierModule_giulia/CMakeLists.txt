# Copyright: 2010 RobotCub Consortium
# Author: Sean Ryan Fanello, Carlo Ciliberto, Giulia Pasquale
# CopyPolicy: Released under the terms of the GNU GPL v2.0.
cmake_minimum_required(VERSION 2.6)

set(projectname linearClassifierModule_giulia)
project(${projectname})

include(FindLibSVMLin.cmake)

find_package(Gurls QUIET)

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${ICUB_INCLUDE_DIRS})

#add common sources

set(folder_header ${folder_header} include/linearClassifierThread.h
                                   include/linearClassifierModule.h
                                   include/win_dirent.h ) 
								   
set(folder_source ${folder_source} src/linearClassifierThread.cpp
                                   src/linearClassifierModule.cpp
                                   src/main.cpp )
								   
#add library specific sources

IF(Gurls_FOUND)
  add_definitions(-DGURLS_AVAILABLE=1)
  add_definitions(${Gurls_DEFINITIONS})
  set(folder_header ${folder_header} include/RLSLinear.h)                                    
  set(folder_source ${folder_source} src/RLSLinear.cpp)  
  include_directories(${Gurls_INCLUDE_DIRS})
ENDIF()

IF(LIBSVMLIN_FOUND)
   add_definitions(-DLIBSVMLIN_AVAILABLE=1)
   set(folder_header ${folder_header} include/SVMLinear.h)                                    
   set(folder_source ${folder_source} src/SVMLinear.cpp)    
   include_directories(${LIBSVMLIN_INCLUDE_DIR})                    
ENDIF()

#if at least one learning library has been found create executable
if (LIBSVMLIN_FOUND OR Gurls_FOUND)
   source_group("Source Files" FILES ${folder_source})
   source_group("Header Files" FILES ${folder_header})
   include_directories(${PROJECT_SOURCE_DIR}/include)
   add_executable(${projectname} ${folder_source} ${folder_header} )
endif()

target_link_libraries(${projectname} ${YARP_LIBRARIES})

IF(LIBSVMLIN_FOUND)
  target_link_libraries(${projectname} ${LIBSVMLIN_LIBRARIES})
ENDIF()

IF(Gurls_FOUND)
  target_link_libraries(${projectname} ${Gurls_LIBRARIES})
ENDIF()

install(TARGETS ${projectname} DESTINATION bin)


