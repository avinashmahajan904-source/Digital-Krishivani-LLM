"""
FastAPI wrapper around the existing `analyze_image()` function.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import importlib.util
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("image_analysis_api")

BASE_DIR = Path(__file__).resolve().parent
CONVERTER_PATH = BASE_DIR / "converter.py"
if not CONVERTER_PATH.exists():
    LEGACY_CONVERTER_PATH = BASE_DIR / "conveter.py"
    if LEGACY_CONVERTER_PATH.exists():
        CONVERTER_PATH = LEGACY_CONVERTER_PATH
    else:
        raise FileNotFoundError(
            f"converter.py not found at expected path: {CONVERTER_PATH}. "
            "Check your project structure."
        )

_spec = importlib.util.spec_from_file_location("converter_module", CONVERTER_PATH)
converter_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter_module)

logger.info("Loaded converter module from: %s", CONVERTER_PATH)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "image_analysis_uploads"
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Image Analysis API",
    description="Wraps analyze_image() (Groq + langchain) as an HTTP API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str


class AnalyzeResponse(BaseModel):
    result: dict[str, Any]


def _validate_upload(upload: UploadFile) -> None:
    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded.",
        )

    ext = Path(upload.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


def _save_upload_to_tmp(upload: UploadFile) -> Path:
    ext = Path(upload.filename).suffix.lower()
    tmp_path = UPLOAD_TMP_DIR / f"{uuid.uuid4().hex}{ext}"

    size = 0
    try:
        with tmp_path.open("wb") as out_file:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File too large. Max allowed size is "
                            f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                out_file.write(chunk)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc
    finally:
        upload.file.close()

    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return tmp_path


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(image: UploadFile = File(...)) -> AnalyzeResponse:
    """Accepts an image upload, runs it through analyze_image(), and returns the parsed JSON result."""
    _validate_upload(image)
    tmp_path = _save_upload_to_tmp(image)

    logger.info("Saved upload to %s (original name: %s)", tmp_path, image.filename)

    try:
        result = await run_in_threadpool(
            converter_module.analyze_image,
            prompt_text=None,
            image_path=str(tmp_path),
        )
    except SystemExit as exc:
        logger.error("Model call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model analysis failed: {exc}",
        ) from exc
    except FileNotFoundError as exc:
        logger.error("Image file missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during image analysis.",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return AnalyzeResponse(result=result)
