#!/usr/bin/env python3

import os
import sys
import time
import uuid
import threading
import sqlite3
import gc
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

IMAGE_GENERATION_METHOD = os.environ.get('IMAGE_GENERATION_METHOD', 'huggingface_cpu')
IMAGE_GENERATION_AVAILABLE = False

IMAGE_OUTPUT_DIR = "generated_images"
MAX_IMAGE_CACHE = 100
IMAGE_STEPS = 18
IMAGE_CFG = 6.5
IMAGE_WIDTH = 896
IMAGE_HEIGHT = 896
JUGGERNAUT_MODEL_PATH = "JuggernautXL_Ragnarok_fp16.safetensors"

class StopGenerationException(Exception):
    pass

class PhotoGenerator:
    def __init__(self):
        self.image_pipeline = None
        self.image_lock = asyncio.Lock()
        self.image_initialized = False
        self.torch = None
        self.StableDiffusionXLPipeline = None
        self.EulerAncestralDiscreteScheduler = None
        self.Image = None
        self.current_progress = 0
        self.total_steps = IMAGE_STEPS
        self.progress_callback = None
        self.generation_start_time = None
        self.stop_generation = False
        self.compel = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image_gen")
        
    def lazy_import_image_libs(self):
        global IMAGE_GENERATION_AVAILABLE
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['XFORMERS_MORE_DETAILS'] = '0'
            os.environ['XFORMERS_DISABLED'] = '1'
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            cpu_count = min(16, os.cpu_count() or 16)
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
            os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
            os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
            os.environ['KMP_BLOCKTIME'] = '1'
            
            sys.modules['xformers'] = None
            
            import torch
            torch.cuda.is_available = lambda: False
            torch.backends.cudnn.enabled = False
            
            from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
            from PIL import Image
            
            IMAGE_GENERATION_AVAILABLE = True
            print("Image generation libraries loaded successfully")
            
            self.torch = torch
            self.StableDiffusionXLPipeline = StableDiffusionXLPipeline
            self.EulerAncestralDiscreteScheduler = EulerAncestralDiscreteScheduler
            self.Image = Image
            
            self.torch.set_num_threads(cpu_count)
            self.torch.set_num_interop_threads(cpu_count)
            
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
            
            return True
        except ImportError as e:
            print(f"Failed to load image generation dependencies: {e}")
            return False

    def progress_callback_fn(self, pipe, step: int, timestep: int, callback_kwargs):
        if self.stop_generation:
            print("Stop requested - cancelling image generation")
            raise StopGenerationException("Generation stopped by user")
        
        self.current_progress = step + 1
        percentage = int((self.current_progress / self.total_steps) * 100)
        
        elapsed = time.time() - self.generation_start_time
        avg_time_per_step = elapsed / self.current_progress if self.current_progress > 0 else 0
        remaining_steps = self.total_steps - self.current_progress
        eta_seconds = avg_time_per_step * remaining_steps
        
        eta_minutes = int(eta_seconds // 60)
        eta_secs = int(eta_seconds % 60)
        eta_str = f"{eta_minutes}m {eta_secs}s" if eta_minutes > 0 else f"{eta_secs}s"
        
        cpu_usage = psutil.cpu_percent(interval=None)
        thread_count = threading.active_count()
        print(f"Progress: {percentage}% ({self.current_progress}/{self.total_steps}) | ETA: {eta_str} | CPU Usage: {cpu_usage}% | Threads: {thread_count}")
        
        if self.progress_callback:
            try:
                self.progress_callback(self.current_progress, self.total_steps, percentage, eta_str)
            except Exception as e:
                print(f"Progress callback error: {e}")
        
        return callback_kwargs

    def init_image_pipeline(self):
        if self.image_initialized and self.image_pipeline is not None:
            return True
        
        try:
            print("Loading image generation libraries...")
            if not self.lazy_import_image_libs():
                print("ERROR: Failed to import image generation libraries. Check if torch and diffusers are installed.")
                return False

            model_path = os.path.abspath(JUGGERNAUT_MODEL_PATH)
            if not os.path.exists(model_path):
                print(f"Model file not found at: {model_path}")
                alt_paths = [
                    os.path.join(os.getcwd(), JUGGERNAUT_MODEL_PATH),
                    os.path.join(os.path.dirname(__file__), JUGGERNAUT_MODEL_PATH),
                    os.path.expanduser(f"~/{JUGGERNAUT_MODEL_PATH}"),
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"Found model at alternative path: {model_path}")
                        break
                else:
                    print("ERROR: Could not find model file in any of the checked paths. Please ensure the model file exists.")
                    return False

            print(f"Loading JuggernautXL model from {model_path}...")
            start_time = time.time()

            gc.collect()

            print("Loading pipeline components (this may take a while for a large model)...")
            try:
                pipe = self.StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=self.torch.float32,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    load_connected_pipe=False,
                )
            except Exception as e:
                print(f"ERROR: Failed to load model from {model_path}: {e}")
                print("This might be due to corrupted model file or insufficient memory.")
                return False

            print(f"Model loaded in {time.time() - start_time:.2f} seconds")

            pipe.safety_checker = None

            print("Configuring scheduler...")
            try:
                pipe.scheduler = self.EulerAncestralDiscreteScheduler.from_config(
                    pipe.scheduler.config,
                    timestep_spacing="trailing"
                )
            except Exception as e:
                print(f"ERROR: Failed to configure scheduler: {e}")
                return False

            print("Moving pipeline to CPU (this may take a while)...")
            start_time = time.time()
            try:
                pipe = pipe.to("cpu")
            except Exception as e:
                print(f"ERROR: Failed to move pipeline to CPU: {e}")
                print("This might be due to insufficient CPU memory.")
                return False
            print(f"Pipeline moved to CPU in {time.time() - start_time:.2f} seconds")

            print("Optimizing pipeline for CPU...")
            try:
                pipe.unet.to(memory_format=self.torch.channels_last)
                pipe.vae.to(memory_format=self.torch.channels_last)

                if hasattr(pipe.unet, 'set_default_attn_processor'):
                    pipe.unet.set_default_attn_processor()

                pipe.enable_attention_slicing(slice_size=1)
                pipe.enable_vae_slicing()

                if hasattr(pipe, 'enable_vae_tiling'):
                    pipe.enable_vae_tiling()

                pipe.set_progress_bar_config(disable=False)
            except Exception as e:
                print(f"WARNING: Some optimizations failed: {e}")
                print("Pipeline will still work but may be slower.")

            self.image_pipeline = pipe
            os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
            self.image_initialized = True

            gc.collect()

            print("JuggernautXL pipeline ready!")
            return True

        except ImportError as e:
            print(f"ERROR: Missing required library: {e}")
            print("Please install torch, diffusers, and other dependencies.")
            return False
        except FileNotFoundError as e:
            print(f"ERROR: File not found: {e}")
            return False
        except MemoryError as e:
            print(f"ERROR: Insufficient memory: {e}")
            print("Try closing other applications or use a smaller model.")
            return False
        except Exception as e:
            print(f"ERROR: Pipeline initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def init_background(self):
        print("Starting JuggernautXL pipeline initialization in background...")
        success = self.init_image_pipeline()
        if success:
            print("Background JuggernautXL pipeline initialization complete")
        else:
            print("Background JuggernautXL pipeline initialization failed")

    def _generate_image_sync(self, prompt, negative_prompt):
        """Synchronous image generation to run in thread pool"""
        with self.torch.inference_mode():
            print("Starting image generation...")
            result = self.image_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                num_inference_steps=IMAGE_STEPS,
                guidance_scale=IMAGE_CFG,
                generator=self.torch.manual_seed(int(time.time()) % 2147483647),
                callback_on_step_end=self.progress_callback_fn,
            )
        return result

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def stop(self):
        print("Stop image generation requested")
        self.stop_generation = True

    async def generate_image(self, prompt: str, session_id: str = None, progress_callback=None) -> Tuple[bool, str]:
        if not self.image_initialized or self.image_pipeline is not None:
            print("JuggernautXL pipeline not ready, initializing now...")
            if not self.init_image_pipeline():
                return False, "Failed to initialize JuggernautXL image generation pipeline"
        
        self.stop_generation = False
        
        try:
            if not self.torch:
                return False, "Image generation libraries not available"
            
            print(f"Generating image with JuggernautXL: '{prompt}'")
            print(f"Initial CPU usage: {psutil.cpu_percent(interval=1)}%")
            
            self.progress_callback = progress_callback
            self.current_progress = 0
            self.total_steps = IMAGE_STEPS
            self.generation_start_time = time.time()
            
            async with self.image_lock:
                image_id = str(uuid.uuid4())
                filename = f"{image_id}.png"
                filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)

                negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, deformed, worst quality"

                try:
                    # Run synchronous generation in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, self._generate_image_sync, prompt, negative_prompt)

                    image = result.images[0]
                    image.save(filepath, format='PNG', optimize=False, compress_level=1)

                    elapsed = time.time() - self.generation_start_time
                    print(f"JuggernautXL image saved: {filepath} (took {elapsed:.1f}s)")

                except StopGenerationException:
                    print("Image generation stopped by user")
                    self.progress_callback = None
                    self.stop_generation = False
                    return False, "Image generation stopped by user"
                
                self.progress_callback = None
                self.stop_generation = False
                
                try:
                    from stbrain import db_lock, DB_PATH
                    with db_lock:
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO generated_images 
                            (id, prompt, filename, created_at, session_id)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (image_id, prompt, filename, int(time.time()), session_id))
                        conn.commit()
                        conn.close()
                except ImportError:
                    pass
                
                return True, filepath
                
        except StopGenerationException:
            print("Image generation stopped by user")
            self.progress_callback = None
            self.stop_generation = False
            return False, "Image generation stopped by user"
        except Exception as e:
            print(f"JuggernautXL image generation error: {e}")
            import traceback
            traceback.print_exc()
            self.progress_callback = None
            self.stop_generation = False
            return False, f"Failed to generate image with JuggernautXL: {str(e)}"

    async def cleanup_old_images(self):
        try:
            from stbrain import db_lock, DB_PATH
        except ImportError:
            print("Cannot import database from stbrain")
            return
        
        try:
            current_time = int(time.time())
            fifteen_minutes_ago = current_time - 900
            
            with db_lock:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT filename FROM generated_images 
                    WHERE created_at < ?
                ''', (fifteen_minutes_ago,))
                
                old_images = cursor.fetchall()
                
                for (filename,) in old_images:
                    image_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
                    try:
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            print(f"Deleted old image: {filename}")
                    except Exception as e:
                        print(f"Error deleting image {filename}: {e}")
                
                cursor.execute('DELETE FROM generated_images WHERE created_at < ?', (fifteen_minutes_ago,))
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                if deleted_count > 0:
                    print(f"Cleaned up {deleted_count} old images from database")
                    
        except Exception as e:
            print(f"Image cleanup error: {e}")

photo_generator = PhotoGenerator()

# Cleanup on exit
import atexit
atexit.register(lambda: photo_generator.executor.shutdown(wait=False))