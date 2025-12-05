import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
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
        self.root.geometry("1100x750")
        self.root.resizable(True, True)
        
        # Variables
        self.query_image_path = tk.StringVar()
        self.video_categories = {}  # {"category_name": [video_paths]}
        self.current_category = tk.StringVar()
        self.sample_rate = tk.IntVar(value=30)
        self.force_reprocess = tk.BooleanVar(value=False)
        self.search_all_categories = tk.BooleanVar(value=True)
        self.cache_dir = "video_embeddings_cache"
        self.video_history_file = "video_categories.json"
        self.matcher = None
        self.is_processing = False
        
        # Load video categories
        self.load_video_categories()
        
        # Setup UI
        self.setup_ui()
        
        # Load categories into combobox
        self.update_category_list()
        
    def load_video_categories(self):
        """Load previously uploaded videos organized by categories"""
        if Path(self.video_history_file).exists():
            try:
                with open(self.video_history_file, 'r') as f:
                    data = json.load(f)
                    self.video_categories = data.get('categories', {})
                    
                    # Filter out videos that no longer exist
                    for category in list(self.video_categories.keys()):
                        self.video_categories[category] = [
                            v for v in self.video_categories[category] 
                            if Path(v).exists()
                        ]
                        # Remove empty categories
                        if not self.video_categories[category]:
                            del self.video_categories[category]
            except:
                self.video_categories = {}
        else:
            self.video_categories = {}
        
        # Ensure "Default" category exists
        if "Default" not in self.video_categories:
            self.video_categories["Default"] = []
    
    def save_video_categories(self):
        """Save current video categories to file"""
        data = {
            'categories': self.video_categories,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.video_history_file, 'w') as f:
            json.dump(data, indent=2, fp=f)
    
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
        main_frame.rowconfigure(6, weight=1)
        
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
        
        # Category Management Section
        category_frame = ttk.LabelFrame(main_frame, text="2. Manage Video Categories", padding="10")
        category_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        category_frame.columnconfigure(1, weight=1)
        
        ttk.Label(category_frame, text="Category:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.category_combo = ttk.Combobox(category_frame, textvariable=self.current_category, 
                                          state='readonly', width=30)
        self.category_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.category_combo.bind('<<ComboboxSelected>>', self.on_category_changed)
        
        category_btn_frame = ttk.Frame(category_frame)
        category_btn_frame.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        ttk.Button(category_btn_frame, text="New", command=self.create_new_category).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(category_btn_frame, text="Rename", command=self.rename_category).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(category_btn_frame, text="Delete", command=self.delete_category).pack(
            side=tk.LEFT, padx=2)
        
        # Video Section with checkboxes
        video_frame = ttk.LabelFrame(main_frame, text="3. Select Videos to Search", padding="10")
        video_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(2, weight=1)
        
        # Video category info
        self.category_info_label = ttk.Label(video_frame, text="", foreground="blue")
        self.category_info_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Video buttons
        video_btn_frame = ttk.Frame(video_frame)
        video_btn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(video_btn_frame, text="Add Video", command=self.add_video).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(video_btn_frame, text="Add Multiple", 
                  command=self.add_multiple_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_btn_frame, text="Select All", 
                  command=self.select_all_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_btn_frame, text="Deselect All", 
                  command=self.deselect_all_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_btn_frame, text="Remove Selected", 
                  command=self.remove_selected_videos).pack(side=tk.LEFT, padx=5)
        
        # Video list with checkboxes
        video_list_container = ttk.Frame(video_frame)
        video_list_container.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        video_list_container.columnconfigure(0, weight=1)
        video_list_container.rowconfigure(0, weight=1)
        
        self.video_canvas = tk.Canvas(video_list_container, height=180, bg='white', 
                                     highlightthickness=1, highlightbackground='gray')
        scrollbar = ttk.Scrollbar(video_list_container, orient=tk.VERTICAL, 
                                 command=self.video_canvas.yview)
        self.video_scrollable_frame = ttk.Frame(self.video_canvas)
        
        self.video_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.video_canvas.configure(scrollregion=self.video_canvas.bbox("all"))
        )
        
        self.video_canvas.create_window((0, 0), window=self.video_scrollable_frame, anchor="nw")
        self.video_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.video_checkboxes = []
        self.video_checkbox_vars = []
        
        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="4. Settings", padding="10")
        settings_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Sample rate
        settings_row1 = ttk.Frame(settings_frame)
        settings_row1.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(settings_row1, text="Sample Rate:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(settings_row1, from_=5, to=120, 
                   textvariable=self.sample_rate, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(settings_row1, text="(Lower = more accurate, Higher = faster)").pack(
            side=tk.LEFT, padx=5)
        
        # Search options
        settings_row2 = ttk.Frame(settings_frame)
        settings_row2.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        ttk.Checkbutton(settings_row2, text="Search ALL categories", 
                       variable=self.search_all_categories,
                       command=self.on_search_mode_changed).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Checkbutton(settings_row2, text="Force Reprocess (ignore cache)", 
                       variable=self.force_reprocess).pack(side=tk.LEFT)
        
        # Action Buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.match_btn = ttk.Button(action_frame, text="🔍 Find Match", 
                                    command=self.start_matching, style='Accent.TButton')
        self.match_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="📋 View Cache", 
                  command=self.view_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="🗑️ Clear Cache", 
                  command=self.clear_cache).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📊 Category Stats", 
                  command=self.show_category_stats).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, 
                                                      wrap=tk.WORD, state='disabled')
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure tags for colored text
        self.results_text.tag_configure("header", font=('Helvetica', 12, 'bold'))
        self.results_text.tag_configure("success", foreground="green")
        self.results_text.tag_configure("info", foreground="blue")
        self.results_text.tag_configure("warning", foreground="orange")
        self.results_text.tag_configure("category", foreground="purple", font=('Helvetica', 10, 'bold'))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def update_category_list(self):
        """Update the category combobox with available categories"""
        categories = sorted(self.video_categories.keys())
        self.category_combo['values'] = categories
        
        if categories:
            if not self.current_category.get() or self.current_category.get() not in categories:
                self.current_category.set(categories[0])
        
        self.update_video_checkboxes()
    
    def on_category_changed(self, event=None):
        """Handle category selection change"""
        self.update_video_checkboxes()
    
    def on_search_mode_changed(self):
        """Handle search mode change"""
        if self.search_all_categories.get():
            self.category_info_label.config(text="🌐 Searching ALL categories")
        else:
            category = self.current_category.get()
            count = len(self.video_categories.get(category, []))
            self.category_info_label.config(text=f"📁 Category: {category} ({count} videos)")
    
    def create_new_category(self):
        """Create a new category"""
        category_name = simpledialog.askstring(
            "New Category", 
            "Enter category name:",
            parent=self.root
        )
        
        if category_name:
            category_name = category_name.strip()
            if category_name in self.video_categories:
                messagebox.showwarning("Warning", f"Category '{category_name}' already exists!")
            elif category_name:
                self.video_categories[category_name] = []
                self.save_video_categories()
                self.update_category_list()
                self.current_category.set(category_name)
                self.log_message(f"✓ Created category: {category_name}", "success")
    
    def rename_category(self):
        """Rename the current category"""
        old_name = self.current_category.get()
        
        if old_name == "Default":
            messagebox.showwarning("Warning", "Cannot rename the 'Default' category!")
            return
        
        new_name = simpledialog.askstring(
            "Rename Category",
            f"Rename '{old_name}' to:",
            initialvalue=old_name,
            parent=self.root
        )
        
        if new_name and new_name != old_name:
            new_name = new_name.strip()
            if new_name in self.video_categories:
                messagebox.showwarning("Warning", f"Category '{new_name}' already exists!")
            else:
                self.video_categories[new_name] = self.video_categories.pop(old_name)
                self.save_video_categories()
                self.update_category_list()
                self.current_category.set(new_name)
                self.log_message(f"✓ Renamed '{old_name}' to '{new_name}'", "success")
    
    def delete_category(self):
        """Delete the current category"""
        category_name = self.current_category.get()
        
        if category_name == "Default":
            messagebox.showwarning("Warning", "Cannot delete the 'Default' category!")
            return
        
        video_count = len(self.video_categories.get(category_name, []))
        
        response = messagebox.askyesno(
            "Delete Category",
            f"Delete category '{category_name}' and its {video_count} video(s)?\n\n"
            "Videos will not be deleted from disk, only removed from this category."
        )
        
        if response:
            del self.video_categories[category_name]
            self.save_video_categories()
            self.update_category_list()
            self.log_message(f"✓ Deleted category: {category_name}", "success")
    
    def get_current_video_list(self):
        """Get the video list for the current category"""
        category = self.current_category.get()
        return self.video_categories.get(category, [])
    
    def update_video_checkboxes(self):
        """Rebuild the checkbox list for videos in current category"""
        # Clear existing checkboxes
        for widget in self.video_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.video_checkboxes.clear()
        self.video_checkbox_vars.clear()
        
        category = self.current_category.get()
        videos = self.get_current_video_list()
        
        # Update category info
        self.on_search_mode_changed()
        
        if not videos:
            label = ttk.Label(self.video_scrollable_frame, 
                            text="No videos in this category. Click 'Add Video' to get started.",
                            foreground="gray")
            label.grid(row=0, column=0, padx=10, pady=10)
        else:
            # Create checkbox for each video
            for idx, video_path in enumerate(videos):
                var = tk.BooleanVar(value=True)
                self.video_checkbox_vars.append(var)
                
                video_name = Path(video_path).name
                
                # Check if video is cached
                cached_indicator = ""
                if self.matcher and self.matcher.is_video_cached(video_path, self.sample_rate.get()):
                    cached_indicator = " ⚡"
                
                cb = ttk.Checkbutton(
                    self.video_scrollable_frame,
                    text=f"{video_name}{cached_indicator}",
                    variable=var
                )
                cb.grid(row=idx, column=0, sticky=tk.W, padx=5, pady=2)
                self.video_checkboxes.append(cb)
        
        # Update canvas scroll region
        self.video_scrollable_frame.update_idletasks()
        self.video_canvas.configure(scrollregion=self.video_canvas.bbox("all"))
    
    def select_all_videos(self):
        """Select all video checkboxes"""
        for var in self.video_checkbox_vars:
            var.set(True)
    
    def deselect_all_videos(self):
        """Deselect all video checkboxes"""
        for var in self.video_checkbox_vars:
            var.set(False)
    
    def get_selected_videos(self):
        """Get list of selected video paths from current category"""
        if self.search_all_categories.get():
            # Get all videos from all categories
            all_videos = []
            for videos in self.video_categories.values():
                all_videos.extend(videos)
            return list(set(all_videos))  # Remove duplicates
        else:
            # Get only selected videos from current category
            videos = self.get_current_video_list()
            selected = []
            for idx, var in enumerate(self.video_checkbox_vars):
                if var.get() and idx < len(videos):
                    selected.append(videos[idx])
            return selected
    
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.query_image_path.set(filename)
            self.log_message(f"Selected image: {Path(filename).name}", "info")
    
    def add_video(self):
        """Add a single video to current category"""
        category = self.current_category.get()
        
        filename = filedialog.askopenfilename(
            title=f"Add Video to '{category}'",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if filename:
            videos = self.get_current_video_list()
            if filename not in videos:
                videos.append(filename)
                self.video_categories[category] = videos
                self.save_video_categories()
                self.update_video_checkboxes()
                self.log_message(f"Added to '{category}': {Path(filename).name}", "info")
            else:
                messagebox.showinfo("Info", "Video already exists in this category")
    
    def add_multiple_videos(self):
        """Add multiple videos to current category"""
        category = self.current_category.get()
        
        filenames = filedialog.askopenfilenames(
            title=f"Add Videos to '{category}'",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        videos = self.get_current_video_list()
        added_count = 0
        
        for filename in filenames:
            if filename not in videos:
                videos.append(filename)
                added_count += 1
        
        if added_count > 0:
            self.video_categories[category] = videos
            self.save_video_categories()
            self.update_video_checkboxes()
            self.log_message(f"Added {added_count} video(s) to '{category}'", "info")
    
    def remove_selected_videos(self):
        """Remove selected videos from current category"""
        category = self.current_category.get()
        videos = self.get_current_video_list()
        
        selected_indices = [i for i, var in enumerate(self.video_checkbox_vars) if var.get()]
        
        if not selected_indices:
            messagebox.showinfo("Info", "No videos selected to remove")
            return
        
        response = messagebox.askyesno(
            "Confirm Removal",
            f"Remove {len(selected_indices)} video(s) from '{category}'?\n\n"
            "Videos will not be deleted from disk."
        )
        
        if response:
            for idx in sorted(selected_indices, reverse=True):
                if idx < len(videos):
                    video_name = Path(videos[idx]).name
                    videos.pop(idx)
                    self.log_message(f"Removed from '{category}': {video_name}", "info")
            
            self.video_categories[category] = videos
            self.save_video_categories()
            self.update_video_checkboxes()
    
    def show_category_stats(self):
        """Show statistics about all categories"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Category Statistics")
        stats_window.geometry("500x400")
        
        text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert(tk.END, "="*50 + "\n")
        text.insert(tk.END, "VIDEO CATEGORIES STATISTICS\n")
        text.insert(tk.END, "="*50 + "\n\n")
        
        total_videos = 0
        for category, videos in sorted(self.video_categories.items()):
            video_count = len(videos)
            total_videos += video_count
            
            text.insert(tk.END, f"📁 {category}\n")
            text.insert(tk.END, f"   Videos: {video_count}\n")
            
            # Check cached videos
            if self.matcher:
                cached_count = sum(1 for v in videos 
                                 if self.matcher.is_video_cached(v, self.sample_rate.get()))
                text.insert(tk.END, f"   Cached: {cached_count}\n")
            
            text.insert(tk.END, "\n")
        
        text.insert(tk.END, "-"*50 + "\n")
        text.insert(tk.END, f"Total Categories: {len(self.video_categories)}\n")
        text.insert(tk.END, f"Total Videos: {total_videos}\n")
        
        text.configure(state='disabled')
    
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
        
        selected_videos = self.get_selected_videos()
        if not selected_videos:
            if self.search_all_categories.get():
                messagebox.showerror("Error", "No videos found in any category!")
            else:
                messagebox.showerror("Error", "Please select at least one video to search!")
            return False
        
        if not Path(self.query_image_path.get()).exists():
            messagebox.showerror("Error", "Query image file does not exist!")
            return False
        
        for video in selected_videos:
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
        
        thread = threading.Thread(target=self.run_matching)
        thread.daemon = True
        thread.start()
    
    def run_matching(self):
        try:
            if self.matcher is None:
                self.log_message("="*50, "header")
                self.log_message("Initializing AI Model...", "header")
                self.log_message("="*50, "header")
                self.matcher = VideoFrameMatcher(self.cache_dir)
                self.log_message("✓ Model loaded successfully!\n", "success")
                self.root.after(0, self.update_video_checkboxes)
            
            selected_videos = self.get_selected_videos()
            
            self.log_message("="*50, "header")
            if self.search_all_categories.get():
                self.log_message(f"Searching ALL categories ({len(selected_videos)} videos)...", "header")
            else:
                category = self.current_category.get()
                self.log_message(f"Searching category '{category}' ({len(selected_videos)} videos)...", "header")
            self.log_message("="*50, "header")
            
            results = self.matcher.find_matching_video(
                self.query_image_path.get(),
                selected_videos,
                sample_rate=self.sample_rate.get(),
                force_reprocess=self.force_reprocess.get()
            )
            
            self.display_results(results)
            self.root.after(0, self.update_video_checkboxes)
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
        
        # Find which category the best match belongs to
        best_match_category = None
        for category, videos in self.video_categories.items():
            for video in videos:
                if Path(video).name == best_match['video']:
                    best_match_category = category
                    break
            if best_match_category:
                break
        
        self.log_message(f"\n✓ Best Match: {best_match['video']}", "success")
        if best_match_category:
            self.log_message(f"   Category: {best_match_category}", "category")
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
            # Find category for this video
            video_category = None
            for category, videos in self.video_categories.items():
                for video in videos:
                    if Path(video).name == video_name:
                        video_category = category
                        break
                if video_category:
                    break
            
            cache_indicator = "⚡ (cached)" if data.get('from_cache', False) else ""
            category_indicator = f" [{video_category}]" if video_category else ""
            
            self.log_message(f"\n{idx}. {video_name}{category_indicator} {cache_indicator}", "info")
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
            self.update_video_checkboxes()
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