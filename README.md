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