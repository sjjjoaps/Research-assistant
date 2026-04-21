"""
文献管理路由
GET    /documents                    — 列出所有已入库文献（含处理状态）
GET    /documents/{doc_id}/status    — 查询单个文档的详细处理状态
GET    /documents/{doc_id}/citations — 查询文档的引用列表
POST   /documents/ingest-file        — 同步入库单个文件
POST   /documents/ingest-file/start  — 异步启动单个文件入库，便于前端轮询状态
POST   /documents/ingest-directory   — 入库整个目录
DELETE /documents/{doc_id}           — 精确删除文档（只删独占数据，共享实体/关系保留）
"""
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, status

from api.schemas import (
    CitationItem,
    DeleteDocumentResult,
    DocumentCitationsResponse,
    DocumentItem,
    DocumentStatusResponse,
    IngestDirectoryRequest,
    IngestFileRequest,
    IngestFileResult,
    IngestStartResponse,
)
from src.infrastructure.config import settings
from src.storage.database import MetadataDatabase
from src.storage.graph_store import GraphStore
from src.workflows.ingestion_pipeline import IngestionPipeline
from src.storage.document_status_store import DocumentStatus, DocumentStatusStore, generate_doc_id
from src.storage.extraction_cache import compute_file_hash

router = APIRouter(prefix="/documents", tags=["documents"])
_RUNNING_STATUSES = {"pending", "parsing", "chunking", "metadata", "indexing", "graph", "citations", "extracting"}


def _get_status_store() -> DocumentStatusStore:
    store = DocumentStatusStore()
    store.init_db()
    return store


def _prepare_doc_identity(file_path: str) -> tuple[Path, str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return path, generate_doc_id(compute_file_hash(path))


def _run_ingest_file(file_path: str, enable_entity_extraction: bool, enable_modal_extraction: bool = False) -> None:
    pipeline = IngestionPipeline(
        enable_entity_extraction=enable_entity_extraction,
        enable_modal_extraction=enable_modal_extraction,
        # citation/section 开关从 settings 读取，无需前端传参
    )
    try:
        pipeline.ingest_file(file_path)
    finally:
        pipeline.close()


@router.get("", response_model=list[DocumentItem])
def list_documents():
    """返回 SQLite 中所有已入库文献的元数据，附带处理状态和处理统计。"""
    db = MetadataDatabase()
    db.init_db()
    records = db.list_documents()
    db.close()

    status_store = _get_status_store()
    items = []
    for r in records:
        status_data = None
        if r.doc_id:
            status_data = status_store.get(r.doc_id)
        items.append(DocumentItem(
            id=r.id,
            file_path=r.file_path,
            doc_id=r.doc_id,
            title=r.title,
            authors=r.authors,
            institution=r.institution,
            year=r.year,
            abstract=r.abstract,
            keywords=r.keywords,
            status=status_data.status if status_data else None,
            current_step=status_data.current_step if status_data else None,
            chunk_count=len(status_data.chunk_ids) if status_data else 0,
            entity_count=len(status_data.entity_ids) if status_data else 0,
            relation_count=len(status_data.relation_ids) if status_data else 0,
            error_message=status_data.error_message if status_data else None,
            updated_at=status_data.updated_at if status_data else None,
        ))
    status_store.close()
    return items


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
def get_document_status(doc_id: str):
    """查询单个文档的详细处理状态。"""
    status_store = _get_status_store()
    s = status_store.get(doc_id)
    status_store.close()
    if s is None:
        raise HTTPException(status_code=404, detail=f"未找到 doc_id={doc_id} 的状态记录")
    return DocumentStatusResponse(
        doc_id=s.doc_id,
        file_path=s.file_path,
        status=s.status,
        current_step=s.current_step,
        chunk_count=len(s.chunk_ids),
        entity_count=len(s.entity_ids),
        relation_count=len(s.relation_ids),
        error_message=s.error_message,
        updated_at=s.updated_at,
    )


@router.post("/ingest-file", response_model=IngestFileResult)
def ingest_file(req: IngestFileRequest):
    """入库单个文件（PDF / DOCX / TXT）"""
    pipeline = IngestionPipeline(enable_entity_extraction=req.enable_entity_extraction)
    # citation/section/chunking 开关从 settings 读取
    try:
        result = pipeline.ingest_file(req.file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline.close()
    return IngestFileResult(**result)


_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/upload", response_model=IngestStartResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """前端文件上传接口：保存文件到 data/raw_data/ 后异步触发入库。"""
    safe_name = Path(file.filename).name  # strip any path traversal
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {suffix}，允许：{', '.join(_ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件超过大小限制（最大 {_MAX_UPLOAD_BYTES // 1024 // 1024} MB）")

    upload_dir = settings.data_dir / "raw_data"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_name
    dest.write_bytes(content)

    doc_id = generate_doc_id(compute_file_hash(dest))
    background_tasks.add_task(_run_ingest_file, str(dest), True, settings.enable_modal_extraction)
    return IngestStartResponse(
        file_path=str(dest),
        doc_id=doc_id,
        accepted=True,
        status="pending",
        current_step="已上传，等待入库",
        message=f"文件 {safe_name} 已接收，后台入库中",
    )


@router.post(
    "/ingest-file/start",
    response_model=IngestStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_ingest_file(req: IngestFileRequest, background_tasks: BackgroundTasks):
    """异步启动单文件入库，便于前端轮询当前阶段。"""
    try:
        path, doc_id = _prepare_doc_identity(req.file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    status_store = _get_status_store()
    try:
        existing_status = status_store.get(doc_id)
        if existing_status and existing_status.status in _RUNNING_STATUSES:
            return IngestStartResponse(
                file_path=str(path),
                doc_id=doc_id,
                accepted=False,
                status=existing_status.status,
                current_step=existing_status.current_step,
                message="该文档已在处理中，前端可直接轮询当前状态。",
            )

        if existing_status and existing_status.status == "processed":
            return IngestStartResponse(
                file_path=str(path),
                doc_id=doc_id,
                accepted=False,
                status=existing_status.status,
                current_step=existing_status.current_step,
                message="该文档已完成入库，无需重复启动。",
            )

        status_store.upsert(DocumentStatus(
            file_path=str(path),
            doc_id=doc_id,
            status="pending",
            current_step="等待入库",
        ))
    finally:
        status_store.close()

    background_tasks.add_task(_run_ingest_file, str(path), req.enable_entity_extraction)
    return IngestStartResponse(
        file_path=str(path),
        doc_id=doc_id,
        accepted=True,
        status="pending",
        current_step="等待入库",
        message="已启动后台入库任务，可轮询状态接口观察各阶段进度。",
    )


@router.post("/ingest-directory", response_model=list[IngestFileResult])
def ingest_directory(req: IngestDirectoryRequest):
    """入库目录下所有支持格式的文件"""
    pipeline = IngestionPipeline(enable_entity_extraction=req.enable_entity_extraction)
    # citation/section/chunking 开关从 settings 读取
    try:
        results = pipeline.ingest_directory(req.directory)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline.close()
    return [IngestFileResult(**r) for r in results]


@router.delete("/{doc_id}", response_model=DeleteDocumentResult)
def delete_document(doc_id: str):
    """精确删除文档。

    只删除该文档独占的 chunk、向量和实体，被其他文档共享的实体和关系保留。
    """
    pipeline = IngestionPipeline()
    try:
        result = pipeline.delete_document(doc_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0] if e.args else str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline.close()
    return DeleteDocumentResult(**result)


@router.get("/{doc_id}/citations", response_model=DocumentCitationsResponse)
def get_document_citations(doc_id: str):
    """查询文档的引用列表。"""
    status_store = _get_status_store()
    s = status_store.get(doc_id)
    status_store.close()
    if s is None:
        raise HTTPException(status_code=404, detail=f"未找到 doc_id={doc_id}")

    graph_store = GraphStore()
    graph_store.init_schema()
    citations_raw = graph_store.get_document_citations(s.file_path)
    graph_store.close()

    return DocumentCitationsResponse(
        doc_id=doc_id,
        file_path=s.file_path,
        citation_count=len(citations_raw),
        citations=[CitationItem(**c) for c in citations_raw],
    )
