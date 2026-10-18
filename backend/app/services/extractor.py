import io
import os
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
import pypdf
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table

from backend.app.config import settings
from backend.app.services.embedding import chunk_text

@dataclass
class ExtractionResult:
    title: str
    filename: str
    raw_text: str
    chunks: List[str]
    chunk_metadatas: List[Dict[str, Any]]
    file_type: str

def sanitize_filename(filename: Optional[str]) -> str:
    """Treat filename strictly as untrusted display metadata, never as a path."""
    if not filename:
        return "uploaded_document"
    # Strip path separators and directories
    base = os.path.basename(filename).strip()
    # Strip null bytes and control chars
    sanitized = "".join(c for c in base if c.isprintable() and c not in ["\x00", "/", "\\"])
    return sanitized if sanitized else "uploaded_document"

def extract_pdf(file_bytes: bytes, safe_filename: str) -> tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Extracts text and per-page chunks from a PDF.
    Validates PDF header, encryption status, and page limits.
    """
    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file: Missing %PDF- header."
        )

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable PDF file: {str(e)}"
        )

    if reader.is_encrypted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Encrypted or password-protected PDF files are not supported."
        )

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF contains no pages."
        )

    if total_pages > settings.MAX_DOCUMENT_PAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF page count ({total_pages}) exceeds maximum allowed limit of {settings.MAX_DOCUMENT_PAGES} pages."
        )

    page_texts: List[str] = []
    chunks: List[str] = []
    chunk_metadatas: List[Dict[str, Any]] = []
    total_extracted_length = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        try:
            extracted = page.extract_text() or ""
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read page {page_num} of PDF: {str(e)}"
            )

        extracted_clean = extracted.strip()
        if extracted_clean:
            total_extracted_length += len(extracted_clean)
            if total_extracted_length > settings.MAX_EXTRACTED_TEXT_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Extracted document text length exceeds maximum allowed limit of {settings.MAX_EXTRACTED_TEXT_LENGTH} chars."
                )

            page_texts.append(extracted_clean)
            page_chunks = chunk_text(extracted_clean)
            for ch in page_chunks:
                chunks.append(ch)
                if len(chunks) > settings.MAX_DOCUMENT_CHUNKS:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Document generated chunk count exceeding maximum allowed limit of {settings.MAX_DOCUMENT_CHUNKS} chunks."
                    )
                chunk_metadatas.append({
                    "page_number": page_num,
                    "filename": safe_filename,
                })

    full_raw_text = "\n\n".join(page_texts)
    if not full_raw_text.strip():
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="PDF contains no extractable text. Scanned documents or image-only PDFs without an OCR text layer are not supported."
        )

    return full_raw_text, chunks, chunk_metadatas

def extract_docx(file_bytes: bytes, safe_filename: str) -> tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Validates OOXML structure and zip-bomb thresholds, then extracts
    ordered paragraphs and tables from a DOCX document.
    """
    if not file_bytes.startswith(b"PK\x03\x04"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid DOCX file: Missing standard ZIP signature."
        )

    # Validate actual OOXML structure via ZipFile
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
            namelist = zf.namelist()
            # Must contain standard OOXML main document part
            if "word/document.xml" not in namelist:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or malformed DOCX file: Missing word/document.xml OOXML structure."
                )

            # Check uncompressed size against zip-bomb threshold
            max_uncompressed_bytes = settings.MAX_DOCX_UNCOMPRESSED_SIZE_MB * 1024 * 1024
            total_uncompressed = sum(info.file_size for info in zf.infolist())
            if total_uncompressed > max_uncompressed_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"DOCX uncompressed size ({total_uncompressed / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {settings.MAX_DOCX_UNCOMPRESSED_SIZE_MB} MB."
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable DOCX archive: {str(e)}"
        )

    try:
        doc = docx.Document(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse DOCX document: {str(e)}"
        )

    blocks: List[str] = []
    total_extracted_length = 0
    # Preserve exact paragraph and table order using supported iter_inner_content() API
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            p_text = item.text.strip()
            if p_text:
                total_extracted_length += len(p_text)
                if total_extracted_length > settings.MAX_EXTRACTED_TEXT_LENGTH:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Extracted document text length exceeds maximum allowed limit of {settings.MAX_EXTRACTED_TEXT_LENGTH} chars."
                    )
                blocks.append(p_text)
        elif isinstance(item, Table):
            for row in item.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    total_extracted_length += len(row_text)
                    if total_extracted_length > settings.MAX_EXTRACTED_TEXT_LENGTH:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Extracted document text length exceeds maximum allowed limit of {settings.MAX_EXTRACTED_TEXT_LENGTH} chars."
                        )
                    blocks.append(row_text)

    full_raw_text = "\n\n".join(blocks)
    if not full_raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document contains no readable text."
        )

    chunks = chunk_text(full_raw_text)
    if len(chunks) > settings.MAX_DOCUMENT_CHUNKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document generated {len(chunks)} chunks, exceeding maximum allowed limit of {settings.MAX_DOCUMENT_CHUNKS} chunks."
        )

    chunk_metadatas = [{"page_number": None, "filename": safe_filename} for _ in chunks]
    return full_raw_text, chunks, chunk_metadatas

def extract_txt(file_bytes: bytes, safe_filename: str) -> tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Decodes TXT strictly as UTF-8 (allowing UTF-8 BOM). Rejects non-UTF-8 encodings.
    """
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file encoding: text files must be strictly UTF-8 encoded (optional UTF-8 BOM allowed). Error: {str(e)}"
        )

    full_raw_text = text.strip()
    if not full_raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document contains no readable text."
        )

    if len(full_raw_text) > settings.MAX_EXTRACTED_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extracted document text length ({len(full_raw_text)} chars) exceeds maximum allowed limit of {settings.MAX_EXTRACTED_TEXT_LENGTH} chars."
        )

    chunks = chunk_text(full_raw_text)
    if len(chunks) > settings.MAX_DOCUMENT_CHUNKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document generated {len(chunks)} chunks, exceeding maximum allowed limit of {settings.MAX_DOCUMENT_CHUNKS} chunks."
        )

    chunk_metadatas = [{"page_number": None, "filename": safe_filename} for _ in chunks]
    return full_raw_text, chunks, chunk_metadatas

def process_uploaded_document(
    file_bytes: bytes,
    original_filename: Optional[str],
    title: Optional[str] = None
) -> ExtractionResult:
    """
    Validates uploaded file size and extension, dispatches to appropriate extractor,
    and validates post-extraction length and chunk limits.
    """
    # 1. Enforce upload size limit
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Uploaded file size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    safe_filename = sanitize_filename(original_filename)
    _, ext = os.path.splitext(safe_filename)
    ext_lower = ext.lower()

    if ext_lower not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: .pdf, .docx, .txt"
        )

    file_type = ext_lower[1:]  # "pdf", "docx", "txt"

    if file_type == "pdf":
        raw_text, chunks, chunk_metadatas = extract_pdf(file_bytes, safe_filename)
    elif file_type == "docx":
        raw_text, chunks, chunk_metadatas = extract_docx(file_bytes, safe_filename)
    else:  # txt
        raw_text, chunks, chunk_metadatas = extract_txt(file_bytes, safe_filename)

    # Post-extraction limits
    if len(raw_text) > settings.MAX_EXTRACTED_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extracted document text length ({len(raw_text)} chars) exceeds maximum allowed limit of {settings.MAX_EXTRACTED_TEXT_LENGTH} chars."
        )

    if len(chunks) > settings.MAX_DOCUMENT_CHUNKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document generated {len(chunks)} chunks, exceeding maximum allowed limit of {settings.MAX_DOCUMENT_CHUNKS} chunks."
        )

    resolved_title = title.strip() if title and title.strip() else os.path.splitext(safe_filename)[0]

    return ExtractionResult(
        title=resolved_title,
        filename=safe_filename,
        raw_text=raw_text,
        chunks=chunks,
        chunk_metadatas=chunk_metadatas,
        file_type=file_type,
    )
