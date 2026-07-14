import sys
from unittest.mock import MagicMock

# Mock heavy dependencies that might not be installed locally
sys.modules['easyocr'] = MagicMock()
sys.modules['vietocr'] = MagicMock()
sys.modules['vietocr.tool.predictor'] = MagicMock()
sys.modules['vietocr.tool.config'] = MagicMock()
