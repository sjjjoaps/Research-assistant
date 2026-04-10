"""
SQLite 元数据存储模块
使用 SQLAlchemy 2.0 ORM 管理论文元数据的增删查
"""
from pathlib import Path

from sqlalchemy import Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import settings
from src.metadata_extractor import DocumentMetadata


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    authors: Mapped[str] = mapped_column(Text, default="")
    institution: Mapped[str | None] = mapped_column(String(512), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[str] = mapped_column(Text, default="")


class MetadataDatabase:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def add_document(self, file_path: str, metadata: DocumentMetadata) -> int:
        with self.session_factory() as session:
            record = DocumentRecord(
                file_path=file_path,
                title=metadata.title,
                authors="; ".join(metadata.authors),
                institution=metadata.institution,
                year=metadata.year,
                abstract=metadata.abstract,
                keywords="; ".join(metadata.keywords),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def get_document(self, file_path: str) -> DocumentRecord | None:
        with self.session_factory() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.file_path == file_path)
            return session.scalar(stmt)

    def list_documents(self) -> list[DocumentRecord]:
        with self.session_factory() as session:
            stmt = select(DocumentRecord).order_by(DocumentRecord.id)
            return list(session.scalars(stmt).all())

    def delete_document(self, file_path: str) -> bool:
        with self.session_factory() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.file_path == file_path)
            record = session.scalar(stmt)
            if record is None:
                return False

            session.delete(record)
            session.commit()
            return True
    def close(self):
        self.engine.dispose()