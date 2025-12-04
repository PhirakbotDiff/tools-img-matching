import cv2
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import pickle
import hashlib
import json
from datetime import datetime

# Install required packages:
# pip install opencv-python pillow torch torchvision transformers

from transformers import CLIPProcessor, CLIPModel

class VideoFrameMatcher:
    def __init__(self, cache_dir="video_embeddings_cache"):
        """Initialize CLIP model for image similarity matching"""
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata about cached videos"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save metadata about cached videos"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, indent=2, fp=f)
    
    def get_video_hash(self, video_path):
        """Generate hash for video file to detect changes"""
        video_path = Path(video_path)
        # Use file size and modification time for quick hash
        stat = video_path.stat()
        hash_string = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def get_cache_path(self, video_path, sample_rate):
        """Get cache file path for a video"""
        video_hash = self.get_video_hash(video_path)
        cache_name = f"{video_hash}_sr{sample_rate}.pkl"
        return self.cache_dir / cache_name
    
    def is_video_cached(self, video_path, sample_rate):
        """Check if video embeddings are already cached"""
        video_path = Path(video_path)
        cache_path = self.get_cache_path(video_path, sample_rate)
        
        if not cache_path.exists():
            return False
        
        # Check metadata
        video_hash = self.get_video_hash(video_path)
        if video_hash in self.metadata:
            return True
        
        return False
    
    def save_video_cache(self, video_path, sample_rate, embeddings, frame_count):
        """Save video embeddings to cache"""
        video_path = Path(video_path)
        cache_path = self.get_cache_path(video_path, sample_rate)
        video_hash = self.get_video_hash(video_path)
        
        # Save embeddings
        cache_data = {
            'embeddings': embeddings,
            'frame_count': frame_count,
            'sample_rate': sample_rate
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Update metadata
        self.metadata[video_hash] = {
            'video_name': video_path.name,
            'video_path': str(video_path.absolute()),
            'sample_rate': sample_rate,
            'frame_count': frame_count,
            'cache_file': str(cache_path.name),
            'cached_at': datetime.now().isoformat()
        }
        self.save_metadata()
        
        print(f"✓ Cached embeddings for {video_path.name}")
    
    def load_video_cache(self, video_path, sample_rate):
        """Load cached video embeddings"""
        cache_path = self.get_cache_path(video_path, sample_rate)
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        return cache_data['embeddings'], cache_data['frame_count']
    
    def extract_frames(self, video_path, sample_rate=30):
        """
        Extract frames from video at specified sample rate
        
        Args:
            video_path: Path to video file
            sample_rate: Extract 1 frame every N frames (default: 30 = ~1 per second for 30fps video)
        
        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing {video_path.name}: {total_frames} frames at {fps:.2f} fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames
    
    def get_image_embedding(self, image):
        """Get CLIP embedding for a single image"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def find_matching_video(self, query_image_path, video_paths, sample_rate=30, top_k=5, force_reprocess=False):
        """
        Find which video the query image came from
        
        Args:
            query_image_path: Path to the image to match
            video_paths: List of video file paths
            sample_rate: Frame sampling rate
            top_k: Return top K matching frames for inspection
            force_reprocess: If True, ignore cache and reprocess videos
        
        Returns:
            Dictionary with matching results
        """
        # Load query image
        query_image = Image.open(query_image_path).convert('RGB')
        print(f"\nProcessing query image: {query_image_path}")
        
        # Get query image embedding
        query_embedding = self.get_image_embedding(query_image)
        
        results = {}
        best_match = {"video": None, "similarity": -1, "frame_idx": None}
        
        # Process each video
        for video_path in video_paths:
            video_path = Path(video_path)
            print(f"\n{'='*50}")
            print(f"Analyzing video: {video_path.name}")
            
            # Check cache
            use_cache = False
            if not force_reprocess and self.is_video_cached(video_path, sample_rate):
                print(f"✓ Loading cached embeddings for {video_path.name}")
                embeddings, frame_count = self.load_video_cache(video_path, sample_rate)
                use_cache = True
            else:
                # Extract frames and generate embeddings
                frames = self.extract_frames(video_path, sample_rate)
                
                if not frames:
                    print(f"No frames extracted from {video_path.name}")
                    continue
                
                # Generate embeddings for all frames
                print(f"Generating embeddings for {len(frames)} frames...")
                embeddings = []
                for idx, frame in enumerate(frames):
                    if idx % 50 == 0:
                        print(f"  Processing frame {idx}/{len(frames)}")
                    frame_embedding = self.get_image_embedding(frame)
                    embeddings.append(frame_embedding)
                
                embeddings = np.vstack(embeddings)
                
                # Save to cache
                self.save_video_cache(video_path, sample_rate, embeddings, len(frames))
            
            # Calculate similarities for all frames
            print(f"Calculating similarities...")
            # Matrix multiplication for fast similarity computation
            similarities = np.dot(embeddings, query_embedding.T).flatten()
            
            # Find best match in this video
            max_similarity = float(np.max(similarities))
            max_idx = int(np.argmax(similarities))
            
            # Get top K matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_similarities = [float(similarities[i]) for i in top_indices]
            
            results[video_path.name] = {
                "max_similarity": max_similarity,
                "best_frame_idx": max_idx,
                "avg_similarity": float(np.mean(similarities)),
                "top_k_similarities": top_similarities,
                "total_frames_checked": len(embeddings),
                "from_cache": use_cache
            }
            
            print(f"Best match similarity: {max_similarity:.4f} (frame {max_idx})")
            print(f"Average similarity: {np.mean(similarities):.4f}")
            if use_cache:
                print(f"⚡ Used cached embeddings (faster!)")
            
            # Track overall best match
            if max_similarity > best_match["similarity"]:
                best_match = {
                    "video": video_path.name,
                    "similarity": max_similarity,
                    "frame_idx": max_idx,
                    "frame_time_seconds": (max_idx * sample_rate) / 30  # Approximate
                }
        
        return {
            "best_match": best_match,
            "all_results": results
        }
    
    def list_cached_videos(self):
        """List all cached videos"""
        print("\n" + "="*50)
        print("CACHED VIDEOS")
        print("="*50)
        
        if not self.metadata:
            print("No videos cached yet.")
            return
        
        for video_hash, info in self.metadata.items():
            print(f"\nVideo: {info['video_name']}")
            print(f"  Path: {info['video_path']}")
            print(f"  Frames: {info['frame_count']}")
            print(f"  Sample Rate: {info['sample_rate']}")
            print(f"  Cached: {info['cached_at']}")
    
    def clear_cache(self, video_path=None):
        """Clear cache for specific video or all videos"""
        if video_path:
            video_path = Path(video_path)
            video_hash = self.get_video_hash(video_path)
            
            if video_hash in self.metadata:
                cache_file = self.cache_dir / self.metadata[video_hash]['cache_file']
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[video_hash]
                self.save_metadata()
                print(f"✓ Cleared cache for {video_path.name}")
            else:
                print(f"No cache found for {video_path.name}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            self.metadata = {}
            self.save_metadata()
            print("✓ Cleared all cache")

# Example usage
if __name__ == "__main__":
    # Initialize matcher with cache directory
    matcher = VideoFrameMatcher(cache_dir="video_embeddings_cache")
    
    # Define your paths
    query_image = "captured_frame.PNG"  # Your image to match
    videos = [
        "video1.MOV",
        "video2.MOV",
        "video3.MOV",
        "video4.MOV",
        "video5.MOV",
        "video6.MOV",
        "video7.MOV"
    ]
    
    # Optional: List cached videos
    # matcher.list_cached_videos()
    
    # Optional: Clear cache for a specific video or all
    # matcher.clear_cache("video1.mp4")  # Clear specific video
    # matcher.clear_cache()  # Clear all cache
    
    # Find matching video
    # First run: Will process videos and cache embeddings (slower)
    # Subsequent runs: Will use cached embeddings (much faster!)
    # Set force_reprocess=True to ignore cache and reprocess
    results = matcher.find_matching_video(
        query_image, 
        videos, 
        sample_rate=30,
        force_reprocess=False  # Set to True to ignore cache
    )
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"\nBest Match: {results['best_match']['video']}")
    print(f"Similarity Score: {results['best_match']['similarity']:.4f}")
    print(f"Approximate Time: {results['best_match']['frame_time_seconds']:.2f} seconds")
    
    print("\n\nAll Videos Ranked:")
    sorted_results = sorted(
        results['all_results'].items(),
        key=lambda x: x[1]['max_similarity'],
        reverse=True
    )
    
    for idx, (video_name, data) in enumerate(sorted_results, 1):
        cache_indicator = "⚡ (cached)" if data.get('from_cache', False) else ""
        print(f"{idx}. {video_name} {cache_indicator}")
        print(f"   Max Similarity: {data['max_similarity']:.4f}")
        print(f"   Avg Similarity: {data['avg_similarity']:.4f}")
        print()