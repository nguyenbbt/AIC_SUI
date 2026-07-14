from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from typing import Tuple

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
