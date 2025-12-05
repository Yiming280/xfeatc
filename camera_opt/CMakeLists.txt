cmake_minimum_required(VERSION 3.16)

project(CameraOpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# ---- Build library ----
add_library(CameraOpt SHARED
    src/camera_opt.cpp
)

# Export symbol for Windows DLL
target_compile_definitions(CameraOpt PRIVATE CAMERAOPT_EXPORTS)

target_include_directories(CameraOpt
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/thirdparty/include   # 你的第三方头文件（含 OPTApi.h）
)

target_link_libraries(CameraOpt
    PRIVATE
        ${OpenCV_LIBS}
        OPTSDK       # <-- The imported SDK lib
)

target_compile_definitions(CameraOpt PRIVATE USE_OPENCV)

# 可选安装
install(TARGETS CameraOpt RUNTIME DESTINATION bin)
