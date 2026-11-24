"""Base module for the practical document analysis with LLMs project.

Contains minimal set of functions for processing documents and invoking LLMS
Relies on PyMuPDF for PDF rendering and OpenAI for LLM interactions.

"""

import pymupdf
from openai import OpenAI
from pydantic import BaseModel
import base64
from src.config import DEFAULT_OPENAI_MODEL


def pdf_to_png(pdf_path: str) -> list[bytes]:
    """Render each PDF page as PNG bytes, in page order."""
    with pymupdf.open(pdf_path) as pdf:
        return [page.get_pixmap(dpi=150).tobytes("png") for page in pdf]


def extract_from_doc(
    doc_imgs: list[bytes], extraction_schema: type[BaseModel], role_prompt=None
):
    """Extract structured data from a multi-page document, return as JSON object"""
    client = OpenAI()

    default_role_prompt = (
        "Extract information from the attached document.\n"
        "Strictly follow the provided schema and return the data as JSON.\n\n"
    )

    # add additional instructions to default prompt header
    role_prompt = default_role_prompt + (role_prompt or "")

    # Convert all page images into a list of image content parts
    image_contents = [
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{base64.b64encode(img).decode('ascii')}",
        }
        for img in doc_imgs
    ]

    response = client.responses.parse(
        model=DEFAULT_OPENAI_MODEL,
        input=[
            {"role": "system", "content": role_prompt},
            {
                "role": "user",
                "content": image_contents,
            },
        ],
        text_format=extraction_schema,
    )

    if response.output_parsed is None:
        raise ValueError("No structured data returned")
    return response.output_parsed
