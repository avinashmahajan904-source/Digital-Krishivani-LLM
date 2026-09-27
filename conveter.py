import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_APIKEY")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Create a .env file with: GROQ_API_KEY=your_key_here"
        )
    return api_key


def get_image_data_url(image_path: str | Path | None = None) -> str:
    if image_path is None:
        candidates = [
            _repo_root() / "apple-scab-1-460x263.webp",
            _repo_root() / "sample.webp",
            _repo_root() / "sample.jpg",
            _repo_root() / "sample.png",
        ]
        selected = next((p for p in candidates if p.exists()), None)
        if selected is None:
            raise FileNotFoundError(
                "No image path was provided and no default example image was found in the repo root. "
                "Pass an image path to analyze_image()."
            )
        image_file = selected
    else:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

    mime_type = mimetypes.guess_type(image_file.name)[0] or "application/octet-stream"
    encoded_image = base64.b64encode(image_file.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of model output even if it's wrapped in markdown."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{raw}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model output did not contain a JSON object.")

    parsed.setdefault("other_points", [])
    if not isinstance(parsed["other_points"], list):
        parsed["other_points"] = [str(parsed["other_points"])]

    return parsed


def analyze_image(
    prompt_text: str | None = None,
    image_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    api_key = load_api_key()
    image_data_url = get_image_data_url(image_path=image_path)

    if prompt_text is None:
        prompt_text = """Ye image hai. Isme kya problem dikh rahi hai? Analysis karo Hindi main.

Respond ONLY with valid JSON, no extra text, no markdown code fences, in exactly this shape:
{
  "problem": "<ek line mein kya dikh raha hai>",
  "reason": "<yeh problem kyun ho rahi hai>",
  "solution": "<isko kaise theek karein>",
  "other_points": ["<extra tip 1>", "<extra tip 2>", "..."]
}
"""

    model = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.7,
        max_tokens=500,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    )

    try:
        response = model.invoke([message])
    except Exception as exc:
        raise SystemExit(f"Groq API call failed: {exc}") from exc

    raw = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)

    try:
        return extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Model did not return parseable JSON.\n{exc}\n\nRaw output:\n{raw}") from exc


if __name__ == "__main__":
    result = analyze_image()
    print(json.dumps(result, indent=2, ensure_ascii=False))
