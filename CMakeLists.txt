cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(movie)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Ceres 1.8 REQUIRED)
SET(EXTRA_LIBS /projects/grail/supasornnb/local_robson/lib/libgflags.a /projects/grail/supasornnb/local_robson/lib/libf77blas.so)
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -lpthread -lgfortran")
add_definitions(-fopenmp)
link_directories(/projects/grail/supasornnb/local_robson/lib)
include_directories(${INCLUDE_DIRECTORIES} ${CERES_INCLUDE_DIRS}) 
add_definitions(-std=gnu++11)

message(${CMAKE_SYSTEM_VERSION})
message(${CMAKE_CURRENT_SOURCE_DIR})
#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-D_LINUX_MAC -D_OPENCV -w)

add_executable (nomad nomad.cpp)
target_link_libraries (nomad ceres ${CERES_LIBRARY_DEPENDENCIES} ${EXTRA_LIBS})
