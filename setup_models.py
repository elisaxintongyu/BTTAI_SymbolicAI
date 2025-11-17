"""
Setup script to download OpenCV Zoo models.

Uses huggingface_hub for reliable model downloads from Hugging Face.
"""

import os
import urllib.request
from pathlib import Path
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_huggingface_hub(model_path: Path) -> bool:
    """Try downloading using huggingface_hub library."""
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Using huggingface_hub to download model...")
        
        # Try different possible repo structures
        repo_options = [
            ("opencv/yolox_nano", "yolox_nano.onnx"),
            ("opencv/opencv_zoo", "models/yolox_nano/yolox_nano.onnx"),
        ]
        
        for repo_id, filename in repo_options:
            try:
                logger.info(f"Trying repo: {repo_id}, file: {filename}")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(model_path.parent),
                )
                # Move to expected location
                if downloaded_path != str(model_path):
                    import shutil
                    if Path(downloaded_path).exists():
                        shutil.move(downloaded_path, model_path)
                logger.info(f"✓ Model downloaded successfully to {model_path}")
                return True
            except Exception as e:
                logger.debug(f"  Repo {repo_id} failed: {e}")
                continue
        
        return False
    except ImportError:
        logger.info("huggingface_hub not installed, trying alternative methods...")
        return False
    except Exception as e:
        logger.warning(f"huggingface_hub download failed: {e}")
        return False

def download_with_requests(model_path: Path) -> bool:
    """Try downloading using requests library."""
    try:
        import requests
        # Try different URLs - official YOLOX releases and alternatives
        urls = [
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_nano.onnx",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in urls:
            try:
                logger.info(f"Downloading from: {url}")
                response = requests.get(url, stream=True, timeout=30, headers=headers)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
                
                print()  # New line after progress
                logger.info(f"✓ Model downloaded successfully to {model_path}")
                return True
            except Exception as e:
                logger.warning(f"  URL failed: {e}")
                if model_path.exists():
                    model_path.unlink()
                continue
        
        return False
    except ImportError:
        logger.info("requests not installed, trying urllib...")
        return False
    except Exception as e:
        logger.warning(f"requests download failed: {e}")
        return False

def download_with_urllib(model_path: Path) -> bool:
    """Try downloading using urllib (built-in)."""
    urls = [
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_nano.onnx",
    ]
    
    for url in urls:
        try:
            logger.info(f"Trying: {url}")
            # Use a custom opener with headers
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"✓ Model downloaded successfully to {model_path}")
            return True
        except Exception as e:
            logger.warning(f"  Failed: {e}")
            if model_path.exists():
                model_path.unlink()
            continue
    
    return False

def download_model():
    """Download YOLOX model from OpenCV Zoo (tries multiple methods)."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "yolox_nano.onnx"
    
    if model_path.exists():
        logger.info(f"✓ Model already exists: {model_path}")
        return str(model_path)
    
    logger.info("Downloading YOLOX model from OpenCV Zoo...")
    logger.info("Trying multiple download methods...")
    
    # Try different download methods
    methods = [
        ("huggingface_hub", download_with_huggingface_hub),
        ("requests", download_with_requests),
        ("urllib", download_with_urllib),
    ]
    
    for method_name, method_func in methods:
        try:
            if method_func(model_path):
                return str(model_path)
        except Exception as e:
            logger.warning(f"{method_name} method failed: {e}")
            continue
    
    # If all methods fail, provide manual download instructions
    logger.error("✗ Failed to download from all sources")
    logger.info("\n" + "="*60)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS:")
    logger.info("="*60)
    logger.info("The YOLOX model is optional - the system can work in 'grid' mode without it.")
    logger.info("")
    logger.info("To download manually:")
    logger.info("  1. Visit: https://github.com/Megvii-BaseDetection/YOLOX/releases")
    logger.info("  2. Download 'yolox_nano.onnx' from the latest release")
    logger.info(f"  3. Place it in: {models_dir}")
    logger.info("")
    logger.info("Alternatively, use grid mode (no model needed):")
    logger.info("  python main.py --mode grid --image your_image.png")
    logger.info("="*60)
    
    return None

if __name__ == "__main__":
    download_model()

