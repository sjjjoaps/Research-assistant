from pathlib import Path


PROMPT_FILES = [
    "prompt/keyword_extractor_fallback.md",
    "prompt/citation_extractor_fallback.md",
    "prompt/modal_image_system.md",
    "prompt/modal_image_human.md",
    "prompt/modal_table_system.md",
    "prompt/modal_table_human.md",
    "prompt/description_merger_human.md",
    "prompt/metadata_extractor_human.md",
    "prompt/relation_extra_human.md",
]


def test_phase_51_prompt_files_exist():
    for prompt_file in PROMPT_FILES:
        assert Path(prompt_file).exists(), f"缺少 prompt 文件: {prompt_file}"


def test_llm_modules_load_prompt_files_from_prompt_dir():
    source_checks = {
        "src/retrieval/keyword_extractor.py": "prompt/keyword_extractor_fallback.md",
        "src/ingestion/citation_extractor.py": "prompt/citation_extractor_fallback.md",
        "src/ingestion/modal_processors.py": 'modal_image_system.md',
        "src/ingestion/modal_processors.py#table": 'modal_table_system.md',
        "src/ingestion/description_merger.py": "prompt/description_merger_human.md",
        "src/metadata_extractor.py": "prompt/metadata_extractor_human.md",
        "src/entity_extractor.py": "prompt/relation_extra_human.md",
    }

    for source_ref, needle in source_checks.items():
        source_path = source_ref.split("#", 1)[0]
        content = Path(source_path).read_text(encoding="utf-8")
        assert needle in content, f"{source_path} 未加载 {needle}"
