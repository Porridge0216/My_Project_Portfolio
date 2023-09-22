cmake_minimum_required(VERSION 3.26)
project(projector)

set(CMAKE_CXX_STANDARD 17)
set(JsonCpp_DIR "/opt/homebrew/Cellar/jsoncpp")


# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${projector_SOURCE_DIR}/include)

include_directories(${JsonCpp_DIR}/1.9.5/include)



add_executable(main main.cpp KeystoneCorrection.cpp
        generalFunctions.cpp
        include/KeystoneCorrection.h
        include/generalFunctions.h
)

add_executable(calibration ./calibration/calibration.cpp)




target_link_libraries(main ${OpenCV_LIBS} ${projector_SOURCE_DIR}/lib/libapriltag.a)
file(GLOB LIBRARIES ${JsonCpp_DIR}/1.9.5/lib/lib*)
target_link_libraries(calibration ${OpenCV_LIBS} ${LIBRARIES})

