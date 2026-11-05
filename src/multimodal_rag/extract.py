"""PDF parsing and text chunking helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractedElements:
    """Content extracted from a PDF and the paths of its extracted images."""

    texts: list[str]
    tables: list[str]
    image_paths: list[Path]


def extract_pdf_elements(
    pdf_path: str | Path,
    image_output_dir: str | Path,
) -> tuple[list[Any], ExtractedElements]:
    """Parse a PDF into text, tables, and image files using Unstructured hi_res."""
    source = Path(pdf_path).expanduser().resolve()
    image_dir = Path(image_output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {source.name}")
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise ImportError(
            "PDF parsing requires Unstructured. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    elements = partition_pdf(
        filename=str(source),
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=str(image_dir),
    )

    tables: list[str] = []
    texts: list[str] = []
    for element in elements:
        element_type = type(element).__name__
        if element_type == "Table":
            content = str(element).strip()
            if content:
                tables.append(content)
        elif element_type == "CompositeElement":
            content = str(element).strip()
            if content:
                texts.append(content)

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    extracted = ExtractedElements(texts=texts, tables=tables, image_paths=image_paths)
    return list(elements), extracted


def split_texts(
    texts: list[str],
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split extracted text into token-aware chunks suitable for embedding."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
    if not texts:
        return []

    from langchain_text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text("\n\n".join(text.strip() for text in texts if text.strip()))
