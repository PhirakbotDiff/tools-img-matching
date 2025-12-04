# Video Frame Matcher 🎥🔍

A powerful AI-powered tool that identifies which video a captured image came from. Uses CLIP (Contrastive Language-Image Pre-training) model for accurate visual similarity matching with intelligent caching for lightning-fast repeated queries.

## Features

✨ **Accurate Matching** - Uses state-of-the-art CLIP model for robust visual similarity  
⚡ **Smart Caching** - Process videos once, query multiple times instantly  
🎯 **Detailed Results** - Get similarity scores, frame numbers, and timestamps  
📊 **Batch Processing** - Match against multiple videos simultaneously  
🔧 **Configurable** - Adjust sampling rate for speed vs accuracy tradeoff  
💾 **Cache Management** - Built-in tools to manage cached embeddings

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install opencv-python pillow torch torchvision transformers
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.5.0
pillow>=9.0.0
torch>=1.10.0
torchvision>=0.11.0
transformers>=4.20.0
```

## Quick Start

### Basic Usage

```python
from video_matcher import VideoFrameMatcher

# Initialize the matcher
matcher = VideoFrameMatcher(cache_dir="video_embeddings_cache")

# Define your files
query_image = "captured_frame.jpg"
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

# Find the matching video
results = matcher.find_matching_video(query_image, videos)

# Print the best match
print(f"Best Match: {results['best_match']['video']}")
print(f"Similarity: {results['best_match']['similarity']:.4f}")
```

### First Run vs Cached Run

**First Run** (processes and caches videos):
```
Processing video1.mp4: 9000 frames at 30.00 fps
Extracted 300 frames from video1.mp4
Generating embeddings for 300 frames...
✓ Cached embeddings for video1.mp4
Best match similarity: 0.9542 (frame 145)
```

**Subsequent Runs** (uses cache):
```
✓ Loading cached embeddings for video1.mp4
⚡ Used cached embeddings (faster!)
Best match similarity: 0.9542 (frame 145)
```

## How It Works

1. **Frame Extraction**: Samples frames from each video at configurable intervals
2. **Embedding Generation**: Converts frames to numerical feature vectors using CLIP
3. **Similarity Calculation**: Compares query image against all video frames using cosine similarity
4. **Caching**: Saves embeddings to disk for instant future queries
5. **Results**: Returns ranked matches with similarity scores

## Configuration Options

### Sample Rate

Controls how many frames to extract:

```python
# Extract 1 frame per second (30fps video)
results = matcher.find_matching_video(query_image, videos, sample_rate=30)

# More accurate (slower): 2 frames per second
results = matcher.find_matching_video(query_image, videos, sample_rate=15)

# Faster (less accurate): 1 frame every 2 seconds
results = matcher.find_matching_video(query_image, videos, sample_rate=60)
```

**Recommendation:**
- Start with `sample_rate=30` (1 fps)
- For similar-looking videos, use `sample_rate=15` (2 fps)
- For quick testing, use `sample_rate=60` (0.5 fps)

### Force Reprocessing

Ignore cache and reprocess videos:

```python
results = matcher.find_matching_video(
    query_image, 
    videos, 
    force_reprocess=True
)
```

## Cache Management

### List Cached Videos

```python
matcher.list_cached_videos()
```

Output:
```
==================================================
CACHED VIDEOS
==================================================

Video: video1.mp4
  Path: /path/to/video1.mp4
  Frames: 300
  Sample Rate: 30
  Cached: 2024-12-04T10:30:45.123456
```

### Clear Cache

```python
# Clear cache for specific video
matcher.clear_cache("video1.mp4")

# Clear all cached videos
matcher.clear_cache()
```

### Cache Location

By default, embeddings are stored in `video_embeddings_cache/`:
```
video_embeddings_cache/
├── metadata.json              # Tracks all cached videos
├── a1b2c3d4_sr30.pkl         # Video embeddings
└── e5f6g7h8_sr30.pkl         # Video embeddings
```

## Understanding Results

### Similarity Scores

- **0.95 - 1.00**: Excellent match (very likely the source video)
- **0.90 - 0.95**: Strong match (probably the source video)
- **0.85 - 0.90**: Good match (possibly the source video)
- **< 0.85**: Weak match (unlikely to be the source)

### Result Structure

```python
{
    "best_match": {
        "video": "video2.mp4",
        "similarity": 0.9542,
        "frame_idx": 145,
        "frame_time_seconds": 145.0
    },
    "all_results": {
        "video1.mp4": {
            "max_similarity": 0.7823,
            "best_frame_idx": 89,
            "avg_similarity": 0.6234,
            "top_k_similarities": [0.7823, 0.7654, 0.7432, 0.7321, 0.7156],
            "total_frames_checked": 300,
            "from_cache": true
        },
        "video2.mp4": {
            "max_similarity": 0.9542,
            "best_frame_idx": 145,
            "avg_similarity": 0.7821,
            "top_k_similarities": [0.9542, 0.9234, 0.9102, 0.8987, 0.8876],
            "total_frames_checked": 450,
            "from_cache": false
        }
    }
}
```

## Advanced Usage

### Custom Cache Directory

```python
matcher = VideoFrameMatcher(cache_dir="my_custom_cache")
```

### Multiple Queries

```python
# Process videos once
matcher = VideoFrameMatcher()

# Query multiple images against same videos
for image in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    results = matcher.find_matching_video(image, videos)
    print(f"{image} -> {results['best_match']['video']}")
```

### Integration with Existing Code

```python
from video_matcher import VideoFrameMatcher

def find_video_source(image_path, video_directory):
    """Find which video an image came from"""
    import glob
    
    # Get all videos in directory
    videos = glob.glob(f"{video_directory}/*.mp4")
    
    # Initialize matcher
    matcher = VideoFrameMatcher()
    
    # Find match
    results = matcher.find_matching_video(image_path, videos)
    
    return results['best_match']

# Usage
match = find_video_source("screenshot.jpg", "/path/to/videos")
print(f"Found in: {match['video']} at {match['frame_time_seconds']:.1f}s")
```

## Performance Tips

### Speed Optimization

1. **Use GPU**: Automatically used if available (CUDA-enabled PyTorch)
2. **Adjust Sample Rate**: Higher = faster but less accurate
3. **Leverage Cache**: First run is slow, subsequent runs are instant
4. **Batch Queries**: Process multiple images without reinitializing

### Accuracy Optimization

1. **Lower Sample Rate**: Extract more frames (e.g., `sample_rate=15`)
2. **Higher Resolution**: Ensure query image is clear and high-quality
3. **Similar Conditions**: Best results when query image matches video quality

## Troubleshooting

### "No frames extracted"
- Check video file path is correct
- Ensure video file is not corrupted
- Verify OpenCV can read the video format

### Low similarity scores for correct video
- Try lower sample_rate for more frames
- Check if query image is heavily edited/filtered
- Ensure query image actually came from one of the videos

### Out of memory errors
- Increase sample_rate (extract fewer frames)
- Process videos one at a time
- Use CPU instead of GPU (slower but more memory)

### Cache not working
- Check write permissions for cache directory
- Verify disk space is available
- Use `force_reprocess=True` to regenerate cache

## Technical Details

- **Model**: OpenAI CLIP (ViT-B/32)
- **Similarity Metric**: Cosine similarity
- **Cache Format**: Pickle (.pkl) for embeddings, JSON for metadata
- **Hash Algorithm**: MD5 (file size + modification time)

## Limitations

- Requires the exact frame or visually similar frame to be in the video
- Performance depends on video length and sample rate
- Heavily edited images may not match well
- Does not work with significantly cropped or transformed images

## License

This tool uses the following open-source components:
- CLIP by OpenAI
- PyTorch
- OpenCV
- Transformers by Hugging Face

## Contributing

Suggestions and improvements welcome! Common enhancements:
- Support for more video formats
- Batch processing optimization
- GUI interface
- Real-time video streaming support

## Support

For issues or questions:
1. Check this README for common solutions
2. Verify all dependencies are installed correctly
3. Test with sample videos and images first
4. Check console output for error messages

---

**Happy Matching! 🎬✨**