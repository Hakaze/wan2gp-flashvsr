# Pipeline implementations for FlashVSR plugin
from .base import BasePipeline
from .flashvsr_tiny import FlashVSRTinyPipeline
from .flashvsr_tiny_long import FlashVSRTinyLongPipeline
from .flashvsr_full import FlashVSRFullPipeline

__all__ = [
    'BasePipeline',
    'FlashVSRTinyPipeline',
    'FlashVSRTinyLongPipeline',
    'FlashVSRFullPipeline'
]

