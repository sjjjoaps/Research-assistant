"""
src/retrieval — 检索模块

提供多种检索策略，供 tool_registry.retrieve_knowledge 按 mode 路由：
    LocalRetriever       — local 模式，实体精确检索（GraphRetriever + Semantic）
    GlobalRetriever      — global 模式，关系宏观检索（RelationVectorStore + Graph）
    MixRetriever         — mix 模式，三路融合（Semantic + Local + Global）
    LightRAGDualRetriever — dual 模式，LightRAG 双极检索（Phase 9-1 新增）
                            Low-Level 实体精确 + High-Level 关系宏观 + one-hop 邻居扩展
"""
