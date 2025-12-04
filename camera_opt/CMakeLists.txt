cmake_minimum_required(VERSION 3.18)
project(ai-vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件
set(SOURCE_FILES src/camera_opt.cpp)

# 外部依赖
# set(spdlog_DIR "C:/vcpkg/installed/x64-windows/share/spdlog")
set(CMAKE_PREFIX_PATH 
"C:/vcpkg/installed/x64-windows/share/spdlog"
"C:/vcpkg/installed/x64-windows/share/fmt"
"C:/vcpkg/installed/x64-windows/share/yaml-cpp"
)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# -----------------------------
# 构建共享库
# -----------------------------
add_library(CameraOpt STATIC ${SOURCE_FILES})

# 包含目录
target_include_directories(CameraOpt
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include      # 供外部使用
        ${OpenCV_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(CameraOpt
    PUBLIC
        spdlog::spdlog
        ${OpenCV_LIBS}
        Eigen3::Eigen
        yaml-cpp
        "C:/Users/yiming.wei/source/repos/xfeatc/thirdparty/lib/OPTSDK.lib"       # 如果是外部库确保 find_package 或手动 link path
)

# -----------------------------
# 安装指令
# -----------------------------
install(TARGETS CameraOpt
    EXPORT CameraOptTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
)

# -----------------------------
# 构建测试可执行文件
# -----------------------------
add_executable(opt_live test/live_view.cpp)

# 确保 opt_live 依赖 CameraOpt 先构建
add_dependencies(opt_live CameraOpt)

target_link_libraries(opt_live
    PRIVATE CameraOpt
)

target_include_directories(opt_live
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include   # 头文件路径
)
