cmake_minimum_required(VERSION 3.16)

project(pipeline)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# OpenCV
find_package(OpenCV REQUIRED)

# OpenPose
find_package(OpenPose REQUIRED)

# Add executable
add_executable(pipeline src/main.cpp)

# Link libraries
target_link_libraries(pipeline
    ${OpenCV_LIBS}  
    ${OpenPose_LIBS}                    
)

