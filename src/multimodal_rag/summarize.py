"""Text, table, and image summarization helpers."""

import base64
import mimetypes
from pathlib import Path
from typing import Sequence


def generate_text_summaries(
    texts: Sequence[str],
    tables: Sequence[str],
    *,
    summarize_texts: bool = True,
    model_name: str = "gpt-4o-mini",
    max_concurrency: int = 3,
) -> tuple[list[str], list[str]]:
    """Summarize text and tables while preserving one summary per source item."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least one.")
    if not texts and not tables:
        return [], []

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template(
        "다음 자료를 검색에 활용하기 좋은 한국어로 간결하게 요약하세요. "
        "수치, 기간, 단위, 고유명사는 유지하고 자료에 없는 정보는 추가하지 마세요.\n\n"
        "{element}"
    )
    chain = (
        {"element": lambda item: item}
        | prompt
        | ChatOpenAI(model=model_name, temperature=0)
        | StrOutputParser()
    )
    config = {"max_concurrency": max_concurrency}

    text_items = list(texts)
    if text_items and summarize_texts:
        text_summaries = chain.batch(text_items, config)
    else:
        text_summaries = text_items
    table_summaries = chain.batch(list(tables), config) if tables else []
    return list(text_summaries), list(table_summaries)


def _encode_image(image_path: Path) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError(f"Unsupported image format: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return encoded, mime_type


def generate_image_summaries(
    image_paths: Sequence[str | Path],
    *,
    model_name: str = "gpt-4o-mini",
) -> tuple[list[str], list[str]]:
    """Return data-URI originals and Korean descriptions for extracted images."""
    if not image_paths:
        return [], []

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=model_name, temperature=0, max_tokens=1024)
    prompt = (
        "이 이미지를 검색에 활용할 수 있도록 한국어로 간결하게 설명하세요. "
        "그래프나 표라면 제목, 축, 범례, 핵심 추세와 확인 가능한 수치를 포함하세요. "
        "읽을 수 없는 내용은 추측하지 마세요."
    )
    image_data_uris: list[str] = []
    summaries: list[str] = []

    for item in image_paths:
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        encoded, mime_type = _encode_image(path)
        response = model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ]
                )
            ]
        )
        content = response.content
        if not isinstance(content, str):
            raise TypeError("The vision model returned a non-text summary.")
        image_data_uris.append(f"data:{mime_type};base64,{encoded}")
        summaries.append(content)
    return image_data_uris, summaries
