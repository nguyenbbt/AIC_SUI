from typing import List, Tuple

from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


def _ensure_vietocr_pillow_compatibility() -> None:
    """Restore the Pillow alias still used by VietOCR 0.3.12."""
    if not hasattr(Image, "ANTIALIAS"):
        setattr(Image, "ANTIALIAS", Image.Resampling.LANCZOS)


_ensure_vietocr_pillow_compatibility()


class TextRecognizer:
    """Wrapper around VietOCR to perform text recognition on cropped images."""
    
    def __init__(self, backbone: str = 'vgg_transformer', use_gpu: bool = True):
        """
        Initializes the VietOCR recognizer.
        
        Args:
            backbone (str): VietOCR backbone to use ('vgg_transformer' or 'vgg_seq2seq').
            use_gpu (bool): Whether to use GPU for recognition.
        """
        self.config = Cfg.load_config_from_name(backbone)
        self.config['device'] = 'cuda:0' if use_gpu else 'cpu'
        
        # Initialize predictor
        self.detector = Predictor(self.config)

    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """
        Recognizes text in a cropped image.
        
        Args:
            image (Image.Image): A cropped PIL Image containing a single line of text.
            
        Returns:
            Tuple[str, float]: The recognized text and its confidence probability.
        """
        text, prob = self.detector.predict(image, return_prob=True)
        return text, float(prob)

    def recognize_batch(
        self,
        images: List[Image.Image],
    ) -> List[Tuple[str, float]]:
        """Recognize multiple cropped text regions in one VietOCR call."""
        if not images:
            return []

        texts, probabilities = self.detector.predict_batch(
            images,
            return_prob=True,
        )
        if len(texts) != len(images) or len(probabilities) != len(images):
            raise RuntimeError(
                "VietOCR returned a different number of batch predictions "
                "than input images."
            )
        return [
            (str(text), float(probability))
            for text, probability in zip(texts, probabilities)
        ]
