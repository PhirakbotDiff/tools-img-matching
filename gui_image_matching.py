import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import pickle
import hashlib
import json
from datetime import datetime

class VideoFrameMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Matcher 🎥🔍")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Variables
        self.query_image_path = tk.StringVar()
        self.video_paths = []
        self.sample_rate = tk.IntVar(value=30)
        self.force_reprocess = tk.BooleanVar(value=False)
        self.cache_dir = "video_embeddings_cache"
        self.matcher = None
        self.is_processing = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎥 Video Frame Matcher", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Query Image Section
        query_frame = ttk.LabelFrame(main_frame, text="1. Select Query Image", padding="10")
        query_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        query_frame.columnconfigure(1, weight=1)
        
        ttk.Label(query_frame, text="Image:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(query_frame, textvariable=self.query_image_path, state='readonly').grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(query_frame, text="Browse", command=self.browse_image).grid(
            row=0, column=2, padx=(5, 0))
        
        # Video Section
        video_frame = ttk.LabelFrame(main_frame, text="2. Add Videos to Search", padding="10")
        video_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(1, weight=1)
        
        # Video buttons
        video_btn_frame = ttk.Frame(video_frame)
        video_btn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(video_btn_frame, text="Add Video", command=self.add_video).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(video_btn_frame, text="Add Multiple Videos", 
                  command=self.add_multiple_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_btn_frame, text="Clear All", command=self.clear_videos).pack(
            side=tk.LEFT, padx=5)
        
        # Video listbox
        video_list_frame = ttk.Frame(video_frame)
        video_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_list_frame.columnconfigure(0, weight=1)
        video_list_frame.rowconfigure(0, weight=1)
        
        self.video_listbox = tk.Listbox(video_list_frame, height=6)
        self.video_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(video_list_frame, orient=tk.VERTICAL, 
                                 command=self.video_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        
        ttk.Button(video_list_frame, text="Remove Selected", 
                  command=self.remove_selected_video).grid(row=1, column=0, 
                                                          sticky=tk.W, pady=(5, 0))
        
        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="3. Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(settings_frame, text="Sample Rate:").grid(row=0, column=0, 
                                                            sticky=tk.W, padx=(0, 5))
        sample_spin = ttk.Spinbox(settings_frame, from_=5, to=120, 
                                 textvariable=self.sample_rate, width=10)
        sample_spin.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(settings_frame, text="(Lower = more accurate, Higher = faster)").grid(
            row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Checkbutton(settings_frame, text="Force Reprocess (ignore cache)", 
                       variable=self.force_reprocess).grid(row=1, column=0, 
                                                          columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Action Buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.match_btn = ttk.Button(action_frame, text="🔍 Find Match", 
                                    command=self.start_matching, style='Accent.TButton')
        self.match_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="📋 View Cache", 
                  command=self.view_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Clear Cache", 
                  command=self.clear_cache).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, 
                                                      wrap=tk.WORD, state='disabled')
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure tags for colored text
        self.results_text.tag_configure("header", font=('Helvetica', 12, 'bold'))
        self.results_text.tag_configure("success", foreground="green")
        self.results_text.tag_configure("info", foreground="blue")
        self.results_text.tag_configure("warning", foreground="orange")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.query_image_path.set(filename)
            self.log_message(f"Selected image: {Path(filename).name}", "info")
    
    def add_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename and filename not in self.video_paths:
            self.video_paths.append(filename)
            self.video_listbox.insert(tk.END, Path(filename).name)
            self.log_message(f"Added video: {Path(filename).name}", "info")
    
    def add_multiple_videos(self):
        filenames = filedialog.askopenfilenames(
            title="Select Videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        for filename in filenames:
            if filename not in self.video_paths:
                self.video_paths.append(filename)
                self.video_listbox.insert(tk.END, Path(filename).name)
        if filenames:
            self.log_message(f"Added {len(filenames)} video(s)", "info")
    
    def remove_selected_video(self):
        selection = self.video_listbox.curselection()
        if selection:
            idx = selection[0]
            video_name = self.video_listbox.get(idx)
            self.video_listbox.delete(idx)
            self.video_paths.pop(idx)
            self.log_message(f"Removed video: {video_name}", "info")
    
    def clear_videos(self):
        self.video_listbox.delete(0, tk.END)
        self.video_paths.clear()
        self.log_message("Cleared all videos", "info")
    
    def log_message(self, message, tag=""):
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, message + "\n", tag)
        self.results_text.configure(state='disabled')
        self.results_text.see(tk.END)
    
    def clear_results(self):
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')
    
    def validate_inputs(self):
        if not self.query_image_path.get():
            messagebox.showerror("Error", "Please select a query image!")
            return False
        
        if not self.video_paths:
            messagebox.showerror("Error", "Please add at least one video!")
            return False
        
        if not Path(self.query_image_path.get()).exists():
            messagebox.showerror("Error", "Query image file does not exist!")
            return False
        
        for video in self.video_paths:
            if not Path(video).exists():
                messagebox.showerror("Error", f"Video file does not exist: {Path(video).name}")
                return False
        
        return True
    
    def start_matching(self):
        if not self.validate_inputs():
            return
        
        if self.is_processing:
            messagebox.showwarning("Warning", "Already processing. Please wait...")
            return
        
        self.is_processing = True
        self.match_btn.configure(state='disabled')
        self.progress.start(10)
        self.status_var.set("Processing...")
        self.clear_results()
        
        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=self.run_matching)
        thread.daemon = True
        thread.start()
    
    def run_matching(self):
        try:
            # Initialize matcher if not already done
            if self.matcher is None:
                self.log_message("="*50, "header")
                self.log_message("Initializing AI Model...", "header")
                self.log_message("="*50, "header")
                self.matcher = VideoFrameMatcher(self.cache_dir)
                self.log_message("✓ Model loaded successfully!\n", "success")
            
            # Run matching
            self.log_message("="*50, "header")
            self.log_message("Starting Video Analysis...", "header")
            self.log_message("="*50, "header")
            
            results = self.matcher.find_matching_video(
                self.query_image_path.get(),
                self.video_paths,
                sample_rate=self.sample_rate.get(),
                force_reprocess=self.force_reprocess.get()
            )
            
            # Display results
            self.display_results(results)
            
            self.root.after(0, lambda: self.status_var.set("Matching completed!"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))
            self.log_message(f"\n❌ Error: {str(e)}", "warning")
            self.root.after(0, lambda: self.status_var.set("Error occurred"))
        
        finally:
            self.is_processing = False
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.match_btn.configure(state='normal'))
    
    def display_results(self, results):
        self.log_message("\n" + "="*50, "header")
        self.log_message("🎯 FINAL RESULTS", "header")
        self.log_message("="*50, "header")
        
        best_match = results['best_match']
        self.log_message(f"\n✓ Best Match: {best_match['video']}", "success")
        self.log_message(f"   Similarity Score: {best_match['similarity']:.4f}", "success")
        self.log_message(f"   Frame Index: {best_match['frame_idx']}", "info")
        self.log_message(f"   Approximate Time: {best_match['frame_time_seconds']:.2f} seconds", "info")
        
        self.log_message("\n📊 All Videos Ranked:", "header")
        self.log_message("-"*50)
        
        sorted_results = sorted(
            results['all_results'].items(),
            key=lambda x: x[1]['max_similarity'],
            reverse=True
        )
        
        for idx, (video_name, data) in enumerate(sorted_results, 1):
            cache_indicator = "⚡ (cached)" if data.get('from_cache', False) else ""
            self.log_message(f"\n{idx}. {video_name} {cache_indicator}", "info")
            self.log_message(f"   Max Similarity: {data['max_similarity']:.4f}")
            self.log_message(f"   Avg Similarity: {data['avg_similarity']:.4f}")
            self.log_message(f"   Frames Checked: {data['total_frames_checked']}")
    
    def view_cache(self):
        if self.matcher is None:
            self.matcher = VideoFrameMatcher(self.cache_dir)
        
        cache_window = tk.Toplevel(self.root)
        cache_window.title("Cached Videos")
        cache_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(cache_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert(tk.END, "="*50 + "\n")
        text.insert(tk.END, "CACHED VIDEOS\n")
        text.insert(tk.END, "="*50 + "\n\n")
        
        if not self.matcher.metadata:
            text.insert(tk.END, "No videos cached yet.")
        else:
            for video_hash, info in self.matcher.metadata.items():
                text.insert(tk.END, f"Video: {info['video_name']}\n")
                text.insert(tk.END, f"  Path: {info['video_path']}\n")
                text.insert(tk.END, f"  Frames: {info['frame_count']}\n")
                text.insert(tk.END, f"  Sample Rate: {info['sample_rate']}\n")
                text.insert(tk.END, f"  Cached: {info['cached_at']}\n\n")
        
        text.configure(state='disabled')
    
    def clear_cache(self):
        response = messagebox.askyesno("Clear Cache", 
                                       "Are you sure you want to clear all cached videos?")
        if response:
            if self.matcher is None:
                self.matcher = VideoFrameMatcher(self.cache_dir)
            
            self.matcher.clear_cache()
            messagebox.showinfo("Success", "Cache cleared successfully!")
            self.log_message("✓ Cache cleared", "success")


class VideoFrameMatcher:
    def __init__(self, cache_dir="video_embeddings_cache"):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, indent=2, fp=f)
    
    def get_video_hash(self, video_path):
        video_path = Path(video_path)
        stat = video_path.stat()
        hash_string = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def get_cache_path(self, video_path, sample_rate):
        video_hash = self.get_video_hash(video_path)
        cache_name = f"{video_hash}_sr{sample_rate}.pkl"
        return self.cache_dir / cache_name
    
    def is_video_cached(self, video_path, sample_rate):
        cache_path = self.get_cache_path(video_path, sample_rate)
        if not cache_path.exists():
            return False
        video_hash = self.get_video_hash(video_path)
        return video_hash in self.metadata
    
    def save_video_cache(self, video_path, sample_rate, embeddings, frame_count):
        video_path = Path(video_path)
        cache_path = self.get_cache_path(video_path, sample_rate)
        video_hash = self.get_video_hash(video_path)
        
        cache_data = {
            'embeddings': embeddings,
            'frame_count': frame_count,
            'sample_rate': sample_rate
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.metadata[video_hash] = {
            'video_name': video_path.name,
            'video_path': str(video_path.absolute()),
            'sample_rate': sample_rate,
            'frame_count': frame_count,
            'cache_file': str(cache_path.name),
            'cached_at': datetime.now().isoformat()
        }
        self.save_metadata()
    
    def load_video_cache(self, video_path, sample_rate):
        cache_path = self.get_cache_path(video_path, sample_rate)
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['embeddings'], cache_data['frame_count']
    
    def extract_frames(self, video_path, sample_rate=30):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    
    def find_matching_video(self, query_image_path, video_paths, sample_rate=30, top_k=5, force_reprocess=False):
        query_image = Image.open(query_image_path).convert('RGB')
        query_embedding = self.get_image_embedding(query_image)
        
        results = {}
        best_match = {"video": None, "similarity": -1, "frame_idx": None}
        
        for video_path in video_paths:
            video_path = Path(video_path)
            
            use_cache = False
            if not force_reprocess and self.is_video_cached(video_path, sample_rate):
                embeddings, frame_count = self.load_video_cache(video_path, sample_rate)
                use_cache = True
            else:
                frames = self.extract_frames(video_path, sample_rate)
                if not frames:
                    continue
                
                embeddings = []
                for frame in frames:
                    frame_embedding = self.get_image_embedding(frame)
                    embeddings.append(frame_embedding)
                
                embeddings = np.vstack(embeddings)
                self.save_video_cache(video_path, sample_rate, embeddings, len(frames))
            
            similarities = np.dot(embeddings, query_embedding.T).flatten()
            max_similarity = float(np.max(similarities))
            max_idx = int(np.argmax(similarities))
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
            
            if max_similarity > best_match["similarity"]:
                best_match = {
                    "video": video_path.name,
                    "similarity": max_similarity,
                    "frame_idx": max_idx,
                    "frame_time_seconds": (max_idx * sample_rate) / 30
                }
        
        return {
            "best_match": best_match,
            "all_results": results
        }
    
    def clear_cache(self, video_path=None):
        if video_path:
            video_path = Path(video_path)
            video_hash = self.get_video_hash(video_path)
            if video_hash in self.metadata:
                cache_file = self.cache_dir / self.metadata[video_hash]['cache_file']
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[video_hash]
                self.save_metadata()
        else:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            self.metadata = {}
            self.save_metadata()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFrameMatcherGUI(root)
    root.mainloop()