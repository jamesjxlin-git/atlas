"""Read local documents or uploaded bytes; preserve PDF page provenance."""

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path


@dataclass(frozen=True)
class Document:
    text: str
    source: str


@dataclass
class ParsedUpload:
    filename: str
    fingerprint: str
    documents: list[Document]
    warnings: list[str] = field(default_factory=list)


def parse_upload(filename: str, payload: bytes, *, max_bytes=25 * 1024 * 1024,
                 max_pages=200, max_characters=2_000_000) -> ParsedUpload:
    name = Path(filename.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValueError("Upload a PDF, UTF-8 text file, or Markdown file.")
    if not payload or len(payload) > max_bytes:
        raise ValueError("The file is empty or exceeds the 25 MB upload limit.")
    documents, warnings = [], []
    if suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
        if b"%PDF-" not in payload[:1024]:
            raise ValueError("The uploaded file does not have a PDF header.")
        try:
            reader = PdfReader(io.BytesIO(payload))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("This PDF requires a password.")
            if len(reader.pages) > max_pages:
                raise ValueError(f"PDFs with more than {max_pages} pages are not supported in this prototype.")
            empty_pages, character_count = [], 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                character_count += len(text)
                if character_count > max_characters:
                    raise ValueError("The extracted paper exceeds the text limit.")
                if text.strip():
                    documents.append(Document(text, f"{name}#page={number}"))
                else:
                    empty_pages.append(number)
            if empty_pages:
                warnings.append("No text was extracted from pages " + ", ".join(map(str, empty_pages))
                                + ". Blank pages and scanned content need inspection; OCR is not included.")
        except (PdfReadError, OSError) as error:
            raise ValueError("The PDF could not be read. Try an extractable, unencrypted copy.") from None
    else:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("Text uploads must use UTF-8 encoding.") from None
        if len(text) > max_characters:
            raise ValueError("The paper exceeds the extracted-text limit.")
        if text.strip():
            documents.append(Document(text, name))
    if not documents:
        raise ValueError("No readable text was found. Scanned PDFs require OCR.")
    return ParsedUpload(name, hashlib.sha256(payload).hexdigest(), documents, warnings)


def load_documents(path: str | Path) -> list[Document]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    documents = []
    for file in files:
        if not file.is_file() or file.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        name = file.relative_to(path).as_posix() if path.is_dir() else file.name
        parsed = parse_upload(file.name, file.read_bytes())
        # Preserve relative directory names when loading a corpus from disk.
        for document in parsed.documents:
            suffix = document.source.removeprefix(file.name)
            documents.append(Document(document.text, name + suffix))
    if not documents:
        raise ValueError("No readable .txt, .md, or .pdf documents were found.")
    return documents
