# extraction_cache.py

## 所在层次
存储层 `src/storage/`

## 主体功能
文档解析结果缓存，以 `file_hash + config_hash` 为键缓存解析结果。
文件内容或解析配置发生变化时缓存自动失效。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `compute_file_hash(file_path)` | 计算文件内容 MD5（缓存键的文件部分） |
| `compute_config_hash(config)` | 计算解析配置 MD5（缓存键的配置部分），自定义 encoder 处理 Path/枚举类型 |
| `ExtractionCache.get(file_hash, config_hash)` | 查询缓存，命中返回解析结果，未命中返回 None |
| `ExtractionCache.set(file_hash, config_hash, result)` | 写入缓存 |
| `ExtractionCache.delete(file_hash)` | 删除指定文件的所有缓存记录 |

## 调用关系
- **被调用方**：`ingestion_pipeline`（解析前查询缓存，解析后写入缓存）
- **依赖方**：`SQLAlchemy`、`src/infrastructure/config.py`

## 注意事项
- 缓存键 = `file_hash + config_hash`，同文件不同配置（启用/禁用多模态）不共享缓存
- `config_hash` 使用自定义 `_SafeEncoder` 处理 `Path` 等非 JSON 标准类型
- 缓存存储在 SQLite，与元数据库共用基础设施
