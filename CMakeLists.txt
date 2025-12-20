cmake_minimum_required(VERSION 3.16)

project(pipeline)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# OpenCV
find_package(OpenCV REQUIRED)

# OpenPose
find_package(OpenPose REQUIRED)

# Eigen (header-only)
find_package(Eigen3 REQUIRED)

# Add executable
add_executable(pipeline
    src/main.cpp
    src/VideoLoader.cpp
    src/PoseDetector.cpp
    src/Visualization.cpp
    src/FittingOptimizer.cpp
    src/TemporalSmoother.cpp
    src/SMPLModel.cpp
)

target_include_directories(pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(pipeline
    ${OpenCV_LIBS}  
    ${OpenPose_LIBS}                    
)


add_executable(extract_keypoints
    tools/extract_keypoints.cpp
)

target_link_libraries(extract_keypoints
    ${OpenCV_LIBS}
    openpose
)

target_include_directories(extract_keypoints PRIVATE
    include
)

