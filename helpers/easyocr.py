import easyocr
import sys

# package parameter
this = sys.modules[__name__]
this.reader = None

# init OCR
if this.reader is None:
    this.reader = easyocr.Reader(['th'])


def get_ocr_reader():
    return this.reader
