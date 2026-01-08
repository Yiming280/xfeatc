# XFeatC - C++ Implementation with OPT Camera Integration

A professional C++ port of [XFeat](https://github.com/verlab/accelerated_features) feature detection with integrated OPT camera support for real-time video processing.

## Overview

This repository extends the original XFeat C++ implementation with:
- **Real-time Feature Detection**: Continuous live stream processing with FPS overlay
- **OPT Camera Integration**: Full SDK wrapper with device enumeration and exposure control  
- **Template Matching**: Interactive ROI selection with RANSAC geometry verification
- **Dual-mode Operation**: Static image-pair matching OR live stream matching (no recompilation)
- **Configuration System**: Runtime camera settings via configuration file

## What's New

### OPT Camera Module (`camera_opt/`)
- Device enumeration and frame capture wrapper
- Pixel format conversion (Mono8, BGR8, RGB8)
- Configuration-driven exposure time setting
- Graceful fallback when camera unavailable

### Enhanced Demos
- **DetectDemo**: Feature detection with FPS overlay for static images or live streams
- **MatchDemo**: Template matching with two modes - static image-pair or interactive live stream

### Configuration System
- `camera_opt/camera_config.txt` - Runtime camera settings without recompilation

## Usage

### DetectDemo - Feature Detection

**Static mode** (single image, no camera):
```bash
DetectDemo.exe --model ../../model/xfeat_640x640.onnx --img ../../data/1.png
```

**Live stream mode** (real-time detection from camera):
```bash
DetectDemo.exe --model ../../model/xfeat_640x640.onnx
```

### MatchDemo - Template Matching

**Static mode** (match two images, no camera):
```bash
MatchDemo.exe --model ../../model/xfeat_640x640.onnx --img1 ../../data/1.png --img2 ../../data/2.png
```

**Live stream mode with camera template selection**:
```bash
MatchDemo.exe --model ../../model/xfeat_640x640.onnx
```

**Live stream mode with file template**:
```bash
MatchDemo.exe --model ../../model/xfeat_640x640.onnx --img1 ../../data/1.png
```

## Key Features

- **Dual-mode Operation**: Choose static image matching (no hardware) or live stream (camera required) based on command-line arguments
- **Real-time Metrics**: FPS display, feature count, match statistics, homography confidence
- **Interactive ROI Selection**: Dynamically select template from live frames
- **No Camera Required for Static**: Static image matching works completely offline
- **Configuration-Driven**: Adjust camera exposure via `camera_opt/camera_config.txt`

## Original Work

- **Original XFeat**: https://github.com/verlab/accelerated_features  
- **XFeat C++ Fork**: https://github.com/meyiao/accelerated_features (output format optimization)

## License

- XFeat Core: Apache 2.0
- C++ Implementation & Enhancements: Unlicense (public domain)

## Build as a reusable library

This project now builds the XFeat core as a static library `XFeatLib`. Demo executables link against `XFeatLib` so other projects can consume the library without needing the `.cc`/`.h` sources.

How to build (CMake in VSCode will auto-configure on save):

1. Configure the project in VS Code as usual (the provided `CMakeLists.txt` creates `XFeatLib`).
2. Build the solution/targets from the VS Code CMake UI — `XFeatLib` will be available as a target.

Using `XFeatLib` from another CMake project (example):

```cmake
# In your consuming project's CMakeLists.txt
add_subdirectory(path/to/xfeatc) # optional if you include the repo as a submodule

add_executable(MyApp main.cpp)
target_link_libraries(MyApp PRIVATE XFeatLib CameraOpt)
target_include_directories(MyApp PRIVATE ${PROJECT_SOURCE_DIR}/path/to/xfeatc/src ${PROJECT_SOURCE_DIR}/path/to/xfeatc/camera_opt/include)
```

Minimal `main.cpp` example to call the model:

```cpp
#include "XFeat.h"
#include <opencv2/opencv.hpp>

int main() {
	// create detector (path can be absolute or relative)
	XFeat xfeat("../xfeatc/model/xfeat_640x640.onnx");

	// load grayscale image and resize to 640x640
	cv::Mat img = cv::imread("../xfeatc/data/1.png", cv::IMREAD_GRAYSCALE);
	cv::resize(img, img, cv::Size(640,640));

	std::vector<cv::KeyPoint> keys;
	cv::Mat descs;
	xfeat.DetectAndCompute(img, keys, descs, 1000);

	// draw and show
	cv::Mat color; cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
	cv::drawKeypoints(color, keys, color, cv::Scalar(0,0,255));
	cv::imshow("detect", color);
	cv::waitKey(0);
	return 0;
}
```

Descriptor matching example (use `Matcher` from this repo):

```cpp
// assume descs1, descs2 and keypoints keys1, keys2 have been computed
std::vector<cv::DMatch> matches;
Matcher::Match(descs1, descs2, matches, 0.82f);

// Optional geometric filtering
std::vector<cv::Point2f> pts1, pts2;
for (auto &m : matches) {
	pts1.push_back(keys1[m.queryIdx].pt);
	pts2.push_back(keys2[m.trainIdx].pt);
}
Matcher::RejectBadMatchesF(pts1, pts2, matches, 4.0f);
```

Notes:
- If you prefer to link against a prebuilt `.lib`/`.a` instead of `add_subdirectory`, use `find_library()` / `find_path()` and link the target name `XFeatLib`.
- Demos still compile as executables and demonstrate how to use the API.