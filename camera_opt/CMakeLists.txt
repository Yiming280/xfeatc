cmake_minimum_required(VERSION 3.16)
project(CameraOpt LANGUAGES CXX)

find_package(OpenCV REQUIRED)

set(SRC_FILES
    OptCamera.cpp
)

add_library(CameraOpt STATIC ${SRC_FILES})

target_include_directories(CameraOpt PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Link OpenCV
target_link_libraries(CameraOpt PRIVATE ${OpenCV_LIBS})

# Link the vendor OPTSDK.lib provided under camera_opt/lib (Windows)
if (WIN32)
    set(OPTSDK_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/OPTSDK.lib")
    if (EXISTS ${OPTSDK_LIB})
        target_link_libraries(CameraOpt PRIVATE ${OPTSDK_LIB})
    else()
        message(WARNING "OPTSDK.lib not found at ${OPTSDK_LIB}. Make sure it exists and is the correct file to link.")
    endif()
endif()
