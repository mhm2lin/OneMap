"""
Batch Inference Manager for improving GPU utilization.
Collects inference requests from multiple workers and processes them in batches.
"""
import threading
import queue
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import Future
import torch.multiprocessing as mp


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: int
    image: np.ndarray
    depth: Optional[np.ndarray] = None
    text_query: Optional[str] = None
    future: Future = None


@dataclass
class InferenceResult:
    """Result of model inference."""
    request_id: int
    image_features: torch.Tensor = None
    detections: Dict = None
    text_features: torch.Tensor = None


class BatchInferenceManager:
    """
    Manages batch inference for CLIP and YOLO models.
    Collects requests from multiple workers and processes them in batches
    to maximize GPU utilization.
    """
    
    def __init__(
        self,
        clip_model,
        detector,
        batch_size: int = 4,
        timeout: float = 0.01,
        device: str = "cuda"
    ):
        """
        Args:
            clip_model: CLIP model for feature extraction
            detector: YOLO detector for object detection
            batch_size: Maximum batch size for inference
            timeout: Maximum time to wait for batch to fill (seconds)
            device: Device to run inference on
        """
        self.clip_model = clip_model
        self.detector = detector
        self.batch_size = batch_size
        self.timeout = timeout
        self.device = device
        
        self.request_queue = queue.Queue()
        self.result_dict: Dict[int, InferenceResult] = {}
        self.result_lock = threading.Lock()
        
        self.request_counter = 0
        self.counter_lock = threading.Lock()
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()
        
        print(f"BatchInferenceManager initialized with batch_size={batch_size}")
    
    def _inference_worker(self):
        """Background thread that processes inference requests in batches."""
        while self.running:
            batch_requests: List[InferenceRequest] = []
            batch_images: List[np.ndarray] = []
            
            start_time = time.time()
            
            while len(batch_requests) < self.batch_size:
                try:
                    request = self.request_queue.get(timeout=self.timeout)
                    batch_requests.append(request)
                    batch_images.append(request.image)
                except queue.Empty:
                    break
                
                if time.time() - start_time > self.timeout:
                    break
            
            if not batch_requests:
                continue
            
            try:
                batch_images_np = np.stack(batch_images, axis=0)
                
                with torch.no_grad():
                    batch_features = self.clip_model.get_image_features(batch_images_np)
                
                for i, request in enumerate(batch_requests):
                    result = InferenceResult(
                        request_id=request.request_id,
                        image_features=batch_features[i:i+1] if batch_features is not None else None,
                    )
                    
                    with self.result_lock:
                        self.result_dict[request.request_id] = result
                    
                    if request.future:
                        request.future.set_result(result)
                        
            except Exception as e:
                print(f"Batch inference error: {e}")
                for request in batch_requests:
                    if request.future:
                        request.future.set_exception(e)
    
    def submit_image(self, image: np.ndarray) -> Tuple[int, Future]:
        """
        Submit an image for feature extraction.
        
        Args:
            image: Image array [C, H, W] or [H, W, C]
            
        Returns:
            Tuple of (request_id, future)
        """
        with self.counter_lock:
            self.request_counter += 1
            request_id = self.request_counter
        
        future = Future()
        request = InferenceRequest(
            request_id=request_id,
            image=image,
            future=future
        )
        
        self.request_queue.put(request)
        return request_id, future
    
    def get_result(self, request_id: int, timeout: float = 5.0) -> Optional[InferenceResult]:
        """
        Get result for a specific request.
        
        Args:
            request_id: Request ID returned by submit_image
            timeout: Maximum time to wait for result
            
        Returns:
            InferenceResult or None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.result_lock:
                if request_id in self.result_dict:
                    result = self.result_dict.pop(request_id)
                    return result
            time.sleep(0.001)
        return None
    
    def shutdown(self):
        """Shutdown the inference worker."""
        self.running = False
        self.worker_thread.join(timeout=2.0)


class AsyncModelWrapper:
    """
    Wrapper that provides async batch inference for existing models.
    Can be used as a drop-in replacement for synchronous model calls.
    """
    
    def __init__(self, clip_model, detector, batch_size: int = 4):
        self.clip_model = clip_model
        self.detector = detector
        self.batch_manager = BatchInferenceManager(
            clip_model, detector, batch_size=batch_size
        )
    
    def get_image_features_async(self, image: np.ndarray) -> Tuple[int, Future]:
        """Async version of get_image_features."""
        return self.batch_manager.submit_image(image)
    
    def get_image_features(self, image: np.ndarray) -> torch.Tensor:
        """Synchronous interface that uses batch processing internally."""
        request_id, future = self.batch_manager.submit_image(image)
        result = future.result(timeout=10.0)
        return result.image_features
    
    def detect(self, image: np.ndarray) -> Dict:
        """Run detection (not batched, as YOLO has its own batching)."""
        return self.detector.detect(image)
    
    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        """Get text features (usually cached, no batching needed)."""
        return self.clip_model.get_text_features(texts)
    
    def compute_similarity(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between image and text features."""
        return self.clip_model.compute_similarity(image_feats, text_feats)
    
    def shutdown(self):
        """Shutdown the batch manager."""
        self.batch_manager.shutdown()


class SharedGPUManager:
    """
    Manages GPU memory sharing across multiple processes.
    Uses PyTorch multiprocessing for efficient GPU memory sharing.
    """
    
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.gpu_queues = [mp.Queue() for _ in range(n_workers)]
        self.result_queues = [mp.Queue() for _ in range(n_workers)]
    
    def get_worker_queue(self, worker_id: int) -> mp.Queue:
        """Get the GPU request queue for a specific worker."""
        return self.gpu_queues[worker_id]
    
    def get_result_queue(self, worker_id: int) -> mp.Queue:
        """Get the result queue for a specific worker."""
        return self.result_queues[worker_id]


def optimize_batch_size(
    model,
    input_shape: Tuple[int, int, int] = (3, 640, 640),
    max_batch: int = 32,
    gpu_memory_fraction: float = 0.8
) -> int:
    """
    Automatically determine optimal batch size based on GPU memory.
    
    Args:
        model: The model to optimize for
        input_shape: Input tensor shape (C, H, W)
        max_batch: Maximum batch size to try
        gpu_memory_fraction: Target GPU memory usage fraction
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 1
    
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = total_memory * gpu_memory_fraction
    
    optimal_batch = 1
    for batch_size in [1, 2, 4, 8, 16, 32]:
        if batch_size > max_batch:
            break
        
        try:
            dummy_input = torch.randn(batch_size, *input_shape, device="cuda")
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            current_memory = torch.cuda.memory_allocated()
            
            if current_memory < target_memory:
                optimal_batch = batch_size
            else:
                break
                
            del dummy_input
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            raise e
    
    print(f"Optimal batch size determined: {optimal_batch}")
    return optimal_batch


if __name__ == "__main__":
    from vision_models.clip_dense import ClipModel
    from vision_models.yolo_world_detector import YOLOWorldDetector
    
    print("Testing BatchInferenceManager...")
    
    clip_model = ClipModel("weights/clip.pth", jetson=False)
    detector = YOLOWorldDetector(0.3)
    
    optimal_batch = optimize_batch_size(clip_model.clip_model)
    
    wrapper = AsyncModelWrapper(clip_model, detector, batch_size=optimal_batch)
    
    test_image = np.random.randint(0, 255, (3, 640, 640), dtype=np.uint8)
    
    n_requests = 10
    futures = []
    
    start_time = time.time()
    for i in range(n_requests):
        request_id, future = wrapper.get_image_features_async(test_image)
        futures.append((request_id, future))
    
    results = []
    for request_id, future in futures:
        result = future.result(timeout=10.0)
        results.append(result)
    
    elapsed = time.time() - start_time
    
    print(f"Processed {n_requests} requests in {elapsed:.3f}s")
    print(f"Average time per request: {elapsed/n_requests:.3f}s")
    
    wrapper.shutdown()
    print("Test completed!")
