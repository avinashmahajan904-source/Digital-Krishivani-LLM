import sys
import json

from converter import analyze_image  # same directory — no need for importlib gymnastics


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        image_path = input("Image path dijiye (example: apple-scab-1-460x263.webp): ").strip() or None

    prompt_text = """Ye image hai. Isme kya problem dikh rahi hai? Analysis karo Hindi main.

Respond ONLY with valid JSON, no extra text, no markdown code fences, in exactly this shape:
{
  "problem": "<ek line mein kya dikh raha hai>",
  "reason": "<yeh problem kyun ho rahi hai>",
  "solution": "<isko kaise theek karein>",
  "other_points": ["<extra tip 1>", "<extra tip 2>", "..."]
}
"""

    try:
        result = analyze_image(prompt_text=prompt_text, image_path=image_path)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
