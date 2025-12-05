cmake_minimum_required(VERSION 3.16)
project(XFeatC VERSION 0.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

# onnxruntime
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-1.23.2" CACHE PATH "Path to ONNX Runtime")
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)
#
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Running on Windows")
    set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Running on Linux")
    set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_ROOT}/libonnxruntime.so)
endif ()

# Build CameraOpt library
add_subdirectory(camera_opt)

# --------------------------
# Common libraries / includes
# --------------------------
set(XFEAT_LIBS
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBS}
    CameraOpt
)

set(XFEAT_INCLUDES
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/camera_opt/include
)

# --------------------------
# Source files shared by all demos
# --------------------------
file(GLOB_RECURSE XFEAT_SRC
    src/*.cpp
    src/*.cc
    src/*.hpp
    src/*.h
)

# --------------------------
# Function to create executable
# --------------------------
function(create_xfeat_executable target_name source_file)
    add_executable(${target_name} ${source_file} ${XFEAT_SRC})
    target_link_libraries(${target_name} PRIVATE ${XFEAT_LIBS})
    target_include_directories(${target_name} PUBLIC ${XFEAT_INCLUDES})
endfunction()

# --------------------------
# All demo executables
# --------------------------
create_xfeat_executable(DetectDemo DetectDemo.cc)
create_xfeat_executable(MatchDemo  MatchDemo.cc)
create_xfeat_executable(FlowDemo   FlowDemo.cc)
create_xfeat_executable(testDemo   testDemo.cc)