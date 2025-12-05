cmake_minimum_required(VERSION 3.16)

project(CameraOpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(spdlog_DIR "C:/vcpkg/installed/x64-windows/share/spdlog")
set(fmt_DIR "C:/vcpkg/installed/x64-windows/share/fmt")
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
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
    PUBLIC
        ${OpenCV_LIBS}
        OPTSDK       # <-- The imported SDK lib
        spdlog::spdlog
        fmt::fmt
)

target_compile_definitions(CameraOpt PRIVATE USE_OPENCV)

# 可选安装
install(TARGETS CameraOpt RUNTIME DESTINATION bin)

# ------------------ Build executable ------------------
add_executable(opt_live test/live_view.cpp)

target_include_directories(opt_live PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/thirdparty/include
)
target_link_libraries(opt_live PRIVATE CameraOpt)