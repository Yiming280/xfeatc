cmake_minimum_required(VERSION 3.18)
project(ai-vision)

set(source src/camera_opt.cpp)

set(THIS_PACKAGE_DEPS
  Eigen3
  OpenCV
  spdlog
)

foreach(DEPENDENCY ${THIS_PACKAGE_DEPS})
  find_package(${DEPENDENCY} REQUIRED)
endforeach()


add_library(CameraOpt SHARED
        ${source}
)

target_link_libraries(CameraOpt PUBLIC
        spdlog::spdlog
        ${OpenCV_LIBS}
        OPTSDK
        rt
        pthread
        Eigen3::Eigen
        yaml-cpp
)

set_target_properties(CameraOpt
                      PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


target_include_directories(CameraOpt
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:include
        ${OpenCV_INCLUDE_DIRS}
        ${spdlog_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}/)



add_executable(opt_live test/live_view.cpp)
target_link_libraries(opt_live CameraOpt)
target_include_directories(opt_live
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
)
