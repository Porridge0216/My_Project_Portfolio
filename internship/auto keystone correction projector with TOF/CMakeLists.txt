cmake_minimum_required(VERSION 3.26)
project(tof)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${tof_SOURCE_DIR}/include)


add_executable(calibrate calibration_demo.cpp calibrationTOF.cpp)

add_executable(correct correction_demo.cpp TOFCorrection.cpp
        generalFunctions.cpp)

target_link_libraries(calibrate ${OpenCV_LIBS})
target_link_libraries(correct ${OpenCV_LIBS})