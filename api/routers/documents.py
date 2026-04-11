"""
文献管理路由
GET  /documents              — 列出所有已入库文献
POST /documents/ingest-file  — 入库单个文件
POST /documents/ingest-directory — 入库整个目录
"""
from fastapi import APIRouter, HTTPException

from api.schemas import (
    DocumentItem,
    IngestDirectoryRequest,
    IngestFileRequest,
    IngestFileResult,
)
from src.database import MetadataDatabase
from src.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentItem])
def list_documents():
    """返回 SQLite 中所有已入库文献的元数据"""
    db = MetadataDatabase()
    db.init_db()
    records = db.list_documents()
    db.close()
    return [
        DocumentItem(
            id=r.id,
            file_path=r.file_path,
            title=r.title,
            authors=r.authors,
            institution=r.institution,
            year=r.year,
            abstract=r.abstract,
            keywords=r.keywords,
        )
        for r in records
    ]


@router.post("/ingest-file", response_model=IngestFileResult)
def ingest_file(req: IngestFileRequest):
    """入库单个文件（PDF / DOCX / TXT）"""
    pipeline = IngestionPipeline(enable_entity_extraction=req.enable_entity_extraction)
    try:
        result = pipeline.ingest_file(req.file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline.close()
    return IngestFileResult(**result)


@router.post("/ingest-directory", response_model=list[IngestFileResult])
def ingest_directory(req: IngestDirectoryRequest):
    """入库目录下所有支持格式的文件"""
    pipeline = IngestionPipeline(enable_entity_extraction=req.enable_entity_extraction)
    try:
        results = pipeline.ingest_directory(req.directory)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline.close()
    return [IngestFileResult(**r) for r in results]
