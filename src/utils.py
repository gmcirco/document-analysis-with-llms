"""Utility functions for visualization"""
import pymupdf
from pprint import pprint


def pretty(x):
    pprint(x.model_dump() if hasattr(x, "model_dump") else x, sort_dicts=False)

def extract_single_page(
    pdf_path: str, page_num: int, output_path: str = "page_3.png"
):
  """Extracts a specific page from a PDF and saves it as an image."""
  with pymupdf.open(pdf_path) as doc:
    # PyMuPDF pages are 0-indexed, so page 3 is index 2
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=150)
    pix.save(output_path)