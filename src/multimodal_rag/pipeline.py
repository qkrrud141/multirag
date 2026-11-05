"""Multimodal prompt assembly and answer-chain construction."""

import base64
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

_DATA_URI_PATTERN = re.compile(r"^data:(image/(?:jpeg|png));base64,([A-Za-z0-9+/]+={0,2})$")
_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _content_of(item: Any) -> str:
    if isinstance(item, Document):
        return item.page_content
    if isinstance(item, str):
        return item
    raise TypeError(f"Unsupported retrieved content type: {type(item).__name__}")


def _image_mime_type(content: str) -> str | None:
    match = _DATA_URI_PATTERN.fullmatch(content)
    if match:
        return match.group(1)
    if len(content) >= 100 and _BASE64_PATTERN.fullmatch(content):
        try:
            decoded = base64.b64decode(content, validate=True)
        except ValueError:
            return None
        if decoded.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
    return None


def split_image_text_types(
    documents: Sequence[Any],
) -> dict[str, list[str]]:
    """Separate retrieved image data URIs from text/table source content."""
    images: list[str] = []
    texts: list[str] = []
    for item in documents:
        content = _content_of(item)
        if _image_mime_type(content):
            images.append(content)
        else:
            texts.append(content)
    return {"images": images, "texts": texts}


def build_multimodal_chain(
    retriever,
    *,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 1024,
):
    """Build a chain that searches summaries and answers with retrieved originals."""
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_openai import ChatOpenAI

    def make_prompt(data: dict[str, Any]) -> list[HumanMessage]:
        context = data["context"]
        content: list[dict[str, Any]] = []
        for image in context["images"]:
            mime_type = _image_mime_type(image)
            if mime_type is None:
                raise ValueError("Retrieved image content is not a valid JPEG or PNG.")
            image_url = image
            if not image.startswith("data:"):
                image_url = f"data:{mime_type};base64,{image}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        text_context = "\n\n".join(context["texts"]) or "(검색된 텍스트 자료 없음)"
        content.append(
            {
                "type": "text",
                "text": (
                    "아래 원본 자료를 근거로 질문에 한국어로 답하세요. "
                    "자료에 없는 내용은 추측하지 말고 알 수 없다고 답하세요. "
                    "수치와 단위를 보존하고, 가능하면 근거 자료 종류를 밝혀주세요.\n\n"
                    f"질문: {data['question']}\n\n"
                    f"원본 텍스트 및 표:\n{text_context}"
                ),
            }
        )
        return [HumanMessage(content=content)]

    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(make_prompt)
        | ChatOpenAI(model=model_name, temperature=0, max_tokens=max_tokens)
        | StrOutputParser()
    )
