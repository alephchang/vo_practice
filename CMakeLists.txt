cmake_minimum_required( VERSION 2.8 )
project ( gslam )

set( CMAKE_CXX_COMPILER "g++" )
if(OFF)
    set( CMAKE_BUILD_TYPE "Debug" )
    set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O0" )
    message("Debug mode")
else()
    set( CMAKE_BUILD_TYPE "Release" )
    set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )
    message("Release mode")
endif()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
find_package(Eigen3 REQUIRED)

find_package( OpenCV 3.1 REQUIRED )
find_package(G2O REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Sophus 
find_package(Sophus REQUIRED)

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    g2o_core g2o_stuff g2o_types_sba
    ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so
)
############### dependencies ######################
#gperftools is for performance analysis, remove it if you want
include_directories( ${PROJECT_SOURCE_DIR}/include 
                        ${PROJECT_SOURCE_DIR}
                        "/usr/local/include/gperftools")
add_subdirectory( src )
add_subdirectory( test )
