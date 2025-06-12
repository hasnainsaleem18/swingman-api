# Swingman - Advanced Baseball Swing Analysis System

Swingman is a sophisticated computer vision-based system that provides real-time baseball swing analysis. This module serves as the core CV engine that can be integrated into iOS applications or used standalone.

## Core CV Features

### 1. Real-Time Bat Tracking System
- **Enhanced Motion Detection**: Uses OpenCV-based motion tracking with background subtraction
- **Multi-Method Tracking**: 
  - Primary: YOLO-based bat detection
  - Fallback: Manual tracking with momentum prediction
  - Backup: Motion area detection for reliability
- **Adaptive Frame Processing**: Handles varying lighting conditions and motion speeds
- **60 FPS Processing**: Optimized for real-time performance on mobile devices

### 2. Swing Path Analysis
- **Path Visualization**: Real-time neon trail effect with motion blur
- **Path Metrics**:
  - Swing plane analysis (upward/level/downward)
  - Path consistency scoring
  - Swing speed calculation
  - Follow-through analysis
- **Historical Path Storage**: Records complete swing paths for replay and analysis

### 3. Impact Detection System
- **Multi-Factor Detection**:
  - Brightness change analysis
  - Motion differential calculation
  - Frame-to-frame comparison
- **Impact Point Recording**: Precise location mapping on the bat
- **Sweet Spot Analysis**: Determines if contact was made in the optimal zone

### 4. Pose Analysis Integration
- **MediaPipe Integration**: Full-body pose tracking
- **Baseball-Specific Landmarks**:
  - Head positioning
  - Shoulder alignment
  - Elbow angles
  - Hip rotation
  - Knee flex
  - Ankle stability
- **Stability Scoring**: Real-time analysis of body position stability

### 5. Real-Time Analytics
- **Efficiency Metrics**:
  - Overall swing efficiency score (0-100)
  - Power score based on swing mechanics
  - Path consistency percentage
  - Follow-through completion
  - Pose stability rating
- **Visual Feedback**: Instant metric display with color-coded indicators

## Complete Project Structure

```
project/
├── core/                           # Core CV & ML Components
│   ├── enhanced_swing_tracker.py   # Main tracking orchestrator
│   ├── bat_tracker.py             # Basic bat tracking
│   ├── hybrid_bat_detector.py     # Combined ML+CV bat detection
│   ├── yolo_detector.py          # YOLO model integration
│   ├── pose_analyzer.py           # Body pose analysis
│   ├── impact_detector.py         # Impact detection
│   ├── swing_analyzer.py          # Swing analysis
│   ├── bat_visualizer.py         # Bat visualization
│   ├── bat_grid.py              # Bat grid overlay
│   ├── heatmap_generator.py      # Impact heatmaps
│   ├── swing_data_manager.py     # Data management
│   └── __init__.py               # Module initialization

├── ui/                            # User Interface
│   ├── main_window.py            # Main window controller
│   ├── visualizations.py         # Visual effects
│   └── __init__.py               # UI module init

├── utils/                         # Utilities
│   ├── drawing.py               # Drawing utilities
│   ├── json_encoder.py          # JSON serialization
│   └── __init__.py              # Utils initialization

├── Models/                        # Model Storage
│   └── bat_detection/            # Bat detection models

├── data/                         # Data Storage
│   └── training/                 # Training data

├── tools/                        # Development Tools
│   └── debug/                    # Debugging utilities

├── main.py                       # Main application entry
├── main_swingman.py             # Module entry point
├── train_yolov8.py              # YOLO training script
├── test_ml_modules.py           # ML testing suite
├── swift_bridge.py              # iOS integration bridge
├── setup.py                     # Package setup
├── config.json                  # Configuration
├── demo.py                      # Demo application
├── yolov8n.pt                   # Pre-trained YOLO model
└── requirements.txt             # Dependencies
```

## Detailed File Descriptions

### Core ML & CV Files

1. **enhanced_swing_tracker.py**
   - Primary tracking orchestrator
   - Integrates all CV/ML components
   - Real-time processing pipeline

2. **hybrid_bat_detector.py**
   - Combines traditional CV with ML
   - Fallback detection methods
   - Adaptive tracking strategies

3. **yolo_detector.py**
   - YOLO model integration
   - Object detection pipeline
   - Model inference handling

4. **swing_data_manager.py**
   - Session data management
   - Metrics storage
   - Export functionality

### Utility Files

1. **drawing.py**
   - OpenCV drawing utilities
   - Overlay generation
   - Visual effect helpers

2. **json_encoder.py**
   - Custom JSON serialization
   - Data format conversion
   - Export formatting

### Integration Files

1. **swift_bridge.py**
   - iOS integration layer
   - Data conversion utilities
   - Native bridge functions

2. **setup.py**
   - Package configuration
   - Dependency management
   - Installation scripts

### Testing & Development

1. **train_yolov8.py**
   - YOLO model training
   - Dataset management
   - Training configuration

2. **test_ml_modules.py**
   - ML component testing
   - Performance validation
   - Integration tests

3. **demo.py**
   - Demonstration application
   - Feature showcase
   - Quick testing

### Configuration

1. **config.json**
   - Global settings
   - Model parameters
   - Runtime configuration

2. **yolov8n.pt**
   - Pre-trained YOLO model
   - Base detection weights
   - Transfer learning source

## iOS Integration Guide

### Module Integration

1. **Swift Package Integration**:
```swift
// In your Package.swift
dependencies: [
    .package(url: "your-repo/swingman-cv", from: "1.0.0")
]
```

2. **Module Initialization**:
```swift
import SwingmanCV

// Initialize the tracker
let tracker = SwingmanTracker()

// Configure options
tracker.enablePoseDetection = true
tracker.setVisualizationMode(.minimal)
```

3. **Frame Processing**:
```swift
// Process frames from AVCaptureSession
func captureOutput(_ output: AVCaptureOutput, 
                  didOutput sampleBuffer: CMSampleBuffer,
                  from connection: AVCaptureConnection) {
    guard let frame = convertToMat(sampleBuffer) else { return }
    
    // Process frame
    let results = tracker.processFrame(frame)
    
    // Access results
    let swingPath = results.swingPath
    let metrics = results.metrics
    let impactPoint = results.impactPoint
}
```

### Performance Optimization

- Uses Metal for GPU acceleration where available
- Optimized for iOS devices with A12 Bionic chip or newer
- Memory-efficient frame processing
- Battery-conscious background operation

## Installation

1. **Clone Repository**:
```bash
git clone https://github.com/yourusername/swingman-cv.git
cd swingman-cv
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run Tests**:
```bash
python -m pytest tests/
```

## Usage

### As Standalone Application:
```bash
python main.py --camera 0 --window-size 1280x720
```

### As Python Module:
```python
from swingman.core import EnhancedSwingTracker

# Initialize tracker
tracker = EnhancedSwingTracker(
    enable_pose=True,
    custom_bat_model_path='path/to/model'
)

# Process frames
results = tracker.process_frame(frame)

# Access metrics
metrics = results['metrics']
print(f"Swing Efficiency: {metrics.efficiency_score}%")
```

## Configuration Options

```python
# Tracker configuration
tracker.configure({
    'pose_detection': True,
    'impact_detection': True,
    'visualization_mode': 'full',
    'processing_resolution': (640, 480),
    'min_swing_distance': 30,
    'impact_sensitivity': 0.7
})
```

## Data Export

Exports available in multiple formats:
- JSON: Complete swing data
- CSV: Metrics and coordinates
- MP4: Recorded sessions with overlays
- PNG: Individual frame captures

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary and confidential. All rights reserved.

## Implementation Status

### Current Implementation (MVP - 70% of Pitch Deck)

#### Completed Features (✅)
1. **Core CV System (100%)**
   - Real-time bat tracking
   - Swing path analysis
   - Impact detection
   - Basic metrics calculation

2. **Pose Analysis (80%)**
   - Full body landmark detection
   - Basic pose stability scoring
   - Key baseball mechanics analysis

3. **Analytics Engine (70%)**
   - Efficiency scoring
   - Path consistency
   - Basic power metrics
   - Real-time feedback

4. **UI/Visualization (60%)**
   - Basic swing path visualization
   - Real-time metrics display
   - Simple bat overlay
   - Basic session recording

#### Machine Learning Integration

**Current ML Components:**
1. **YOLO Object Detection**
   - Model: YOLOv8 (ultralytics)
   - Purpose: Bat detection and tracking
   - Location: `core/enhanced_swing_tracker.py`
   - Training: Pre-trained on COCO dataset, fine-tuned for bat detection

2. **Pose Estimation**
   - Model: MediaPipe Pose
   - Purpose: Body landmark detection
   - Location: `core/pose_analyzer.py`
   - Training: Using pre-trained MediaPipe model

3. **Motion Analysis**
   - Type: Custom algorithms
   - Purpose: Swing path analysis
   - Location: `core/swing_analyzer.py`
   - Training: Rule-based system, no training required

### Next Sprint Enhancements (Remaining 30%)

1. **Advanced ML Features**
   - Custom bat detection model training
   - Swing style classification
   - Player-specific model adaptation
   - Multi-angle synthesis

2. **Enhanced Analytics**
   - Predictive power metrics
   - Advanced biomechanics analysis
   - Comparative analysis with pro swings
   - Machine learning-based form correction

3. **UI/UX Improvements**
   - 3D swing visualization
   - Advanced heatmap generation
   - Real-time swing comparison
   - Interactive feedback system

## Detailed File Structure & Purpose

### Core Components (`core/`)

1. **enhanced_swing_tracker.py**
   - Main tracking system orchestrator
   - Integrates all CV and ML components
   - Manages real-time processing pipeline
   - Key classes:
     - `EnhancedSwingTracker`: Main tracking controller
     - `SwingMetrics`: Data structure for metrics

2. **bat_tracker.py**
   - Specialized bat tracking algorithms
   - Multi-method tracking system
   - Motion prediction
   - Key features:
     - Background subtraction
     - Momentum-based prediction
     - Fallback tracking methods

3. **pose_analyzer.py**
   - MediaPipe pose integration
   - Baseball-specific pose analysis
   - Stability scoring system
   - Key components:
     - Landmark detection
     - Pose stability calculation
     - Baseball mechanics analysis

4. **impact_detector.py**
   - Ball-bat impact detection
   - Multi-factor analysis system
   - Impact point recording
   - Features:
     - Brightness analysis
     - Motion differential
     - Sweet spot detection

5. **swing_analyzer.py**
   - Swing mechanics analysis
   - Path consistency calculation
   - Efficiency scoring
   - Components:
     - Path analysis
     - Speed calculation
     - Power estimation

### UI Components (`ui/`)

1. **main_window.py**
   - Main application window
   - Camera feed management
   - User interaction handling
   - Features:
     - Real-time display
     - User controls
     - Session management

2. **visualizations.py**
   - Visual effects renderer
   - Path visualization
   - Metric displays
   - Components:
     - Swing path effects
     - Metric overlays
     - Animation system

### Utility Modules (`utils/`)

1. **cv_utils.py**
   - OpenCV helper functions
   - Image processing utilities
   - Frame manipulation
   - Key utilities:
     - Frame preprocessing
     - Color space conversions
     - Filter applications

2. **data_utils.py**
   - Data processing functions
   - Metric calculations
   - Export utilities
   - Features:
     - Data formatting
     - File I/O
     - Session management

### Entry Points

1. **main.py**
   - Standalone application entry
   - Command-line interface
   - Configuration management
   - Features:
     - Argument parsing
     - System initialization
     - Runtime management

2. **main_swingman.py**
   - Module entry point
   - API definitions
   - Integration interfaces
   - Components:
     - Public API
     - Configuration system
     - Integration utilities

## Model Training & Development

### Current Models

1. **Bat Detection Model**
   - Base: YOLOv8
   - Training data: Custom dataset of bat images
   - Location: `models/bat_detection/`
   - Status: Pre-trained, needs fine-tuning

2. **Pose Model**
   - Base: MediaPipe Pose
   - Purpose: Player pose estimation
   - Location: Using MediaPipe's pre-trained model
   - Status: Production-ready

### Future Model Development

1. **Custom Bat Detection**
   - Dataset collection planned
   - Baseball-specific fine-tuning
   - Multi-angle detection

2. **Swing Classification**
   - Dataset requirements defined
   - Model architecture planned
   - Training pipeline designed

3. **Player Analysis**
   - Biomechanics model planned
   - Comparative analysis system
   - Performance prediction

## Development Roadmap

1. **Sprint 1 (Current - MVP)**
   - Core CV implementation
   - Basic ML integration
   - Essential features

2. **Sprint 2 (Planned)**
   - Advanced ML models
   - Enhanced analytics
   - UI improvements

3. **Sprint 3 (Future)**
   - Custom model training
   - Advanced features
   - Production optimization

## Additional Components Status

### Core ML Components

1. **hybrid_bat_detector.py**
   - Status: ✅ Active
   - Purpose: Combines traditional CV techniques with ML for robust bat detection
   - Features:
     - Multi-method detection fallback
     - Adaptive tracking based on conditions
     - Integration with YOLO and OpenCV trackers

2. **yolo_detector.py**
   - Status: ✅ Active
   - Purpose: Handles YOLO model integration and inference
   - Features:
     - Model loading and inference
     - Detection post-processing
     - Confidence filtering

3. **swing_data_manager.py**
   - Status: ✅ Active
   - Purpose: Manages all session and swing data
   - Features:
     - Session storage and retrieval
     - Metrics calculation and storage
     - Export functionality

### Utility Components

1. **drawing.py**
   - Status: ✅ Active
   - Purpose: Centralized drawing utilities
   - Features:
     - Consistent visual styling
     - Overlay generation
     - Animation effects

2. **json_encoder.py**
   - Status: ✅ Active
   - Purpose: Custom JSON serialization
   - Features:
     - NumPy array serialization
     - Custom object encoding
     - Export formatting

### Model Files

1. **yolov8n.pt**
   - Status: ✅ Active
   - Purpose: Pre-trained YOLO model
   - Usage:
     - Base model for transfer learning
     - Initial bat detection
     - Quick start capability

### Development Tools

1. **train_yolov8.py**
   - Status: ✅ Active
   - Purpose: YOLO model training script
   - Features:
     - Custom dataset training
     - Hyperparameter configuration
     - Model evaluation

2. **test_ml_modules.py**
   - Status: ✅ Active
   - Purpose: Testing ML components
   - Features:
     - Unit tests for ML modules
     - Performance benchmarking
     - Integration testing

### Empty/Placeholder Files (To Be Implemented)

The following files are currently empty and planned for future implementation:

1. **config.json**
   - Status: 🔄 Planned
   - Purpose: Will store global configuration
   - TODO: Implement configuration schema

2. **setup.py**
   - Status: 🔄 Planned
   - Purpose: Package installation setup
   - TODO: Add package metadata and dependencies

3. **demo.py**
   - Status: 🔄 Planned
   - Purpose: Quick feature demonstration
   - TODO: Implement demo scenarios

4. **swift_bridge.py**
   - Status: 🔄 Planned
   - Purpose: iOS integration bridge
   - TODO: Implement native bridging functions