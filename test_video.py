import cv2
import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Install required packages:
# pip install opencv-python pillow torch torchvision transformers

from transformers import CLIPProcessor, CLIPModel

class VideoFrameMatcher:
    def __init__(self):
        """Initialize CLIP model for image similarity matching"""
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
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
    
    def find_matching_video(self, query_image_path, video_paths, sample_rate=30, top_k=5):
        """
        Find which video the query image came from
        
        Args:
            query_image_path: Path to the image to match
            video_paths: List of video file paths
            sample_rate: Frame sampling rate
            top_k: Return top K matching frames for inspection
        
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
            
            # Extract frames
            frames = self.extract_frames(video_path, sample_rate)
            
            if not frames:
                print(f"No frames extracted from {video_path.name}")
                continue
            
            # Calculate similarities for all frames
            similarities = []
            for idx, frame in enumerate(frames):
                frame_embedding = self.get_image_embedding(frame)
                # Cosine similarity
                similarity = np.dot(query_embedding, frame_embedding.T)[0][0]
                similarities.append(similarity)
            
            # Find best match in this video
            max_similarity = max(similarities)
            max_idx = similarities.index(max_similarity)
            
            # Get top K matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_similarities = [similarities[i] for i in top_indices]
            
            results[video_path.name] = {
                "max_similarity": float(max_similarity),
                "best_frame_idx": max_idx,
                "avg_similarity": float(np.mean(similarities)),
                "top_k_similarities": top_similarities,
                "total_frames_checked": len(frames)
            }
            
            print(f"Best match similarity: {max_similarity:.4f} (frame {max_idx})")
            print(f"Average similarity: {np.mean(similarities):.4f}")
            
            # Track overall best match
            if max_similarity > best_match["similarity"]:
                best_match = {
                    "video": video_path.name,
                    "similarity": float(max_similarity),
                    "frame_idx": max_idx,
                    "frame_time_seconds": (max_idx * sample_rate) / 30  # Approximate
                }
        
        return {
            "best_match": best_match,
            "all_results": results
        }

# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = VideoFrameMatcher()
    
    # Define your paths
    # query_image = "captured_frame.PNG"  # Your image to match
    query_image = "captured_frame_2.png"  # Your image to match

    videos = [
        "video1.MOV",
        "video2.MOV",
        "video3.MOV",
        "video4.MOV",
        "video5.MOV"
    ]
    
    # Find matching video
    # sample_rate=30 means 1 frame per second for 30fps video
    # Increase for faster processing, decrease for more accuracy
    results = matcher.find_matching_video(query_image, videos, sample_rate=30)
    
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
        print(f"{idx}. {video_name}")
        print(f"   Max Similarity: {data['max_similarity']:.4f}")
        print(f"   Avg Similarity: {data['avg_similarity']:.4f}")
        print()