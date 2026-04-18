"""
tests/test_description_merger.py

Phase 2.2 验收测试：
1. DescriptionMerger 核心合并逻辑
2. graph_store 合并写入路径（mock Neo4j）
"""
import unittest
from unittest.mock import MagicMock, patch

from src.ingestion.description_merger import DescriptionMerger


# ── 1. DescriptionMerger 核心逻辑 ─────────────────────────────────────────────

class TestDescriptionMergerDeduplicate(unittest.TestCase):

    def setUp(self):
        self.merger = DescriptionMerger(max_direct_chars=2000)

    def test_empty_list(self):
        self.assertEqual(self.merger.merge([]), "")

    def test_none_and_blank_filtered(self):
        self.assertEqual(self.merger.merge(["", "  ", ""]), "")

    def test_single_description(self):
        self.assertEqual(self.merger.merge(["BERT 是预训练语言模型"]), "BERT 是预训练语言模型")

    def test_deduplication(self):
        result = self.merger.merge(["BERT 是预训练语言模型", "BERT 是预训练语言模型"])
        self.assertEqual(result, "BERT 是预训练语言模型")

    def test_deduplication_with_whitespace(self):
        result = self.merger.merge(["BERT 是预训练语言模型  ", "  BERT 是预训练语言模型"])
        self.assertEqual(result, "BERT 是预训练语言模型")

    def test_order_preserved(self):
        descs = ["描述A", "描述B", "描述C"]
        result = self.merger.merge(descs)
        self.assertEqual(result, "描述A\n---\n描述B\n---\n描述C")

    def test_custom_separator(self):
        merger = DescriptionMerger(separator=" | ")
        result = merger.merge(["A", "B"])
        self.assertEqual(result, "A | B")


class TestDescriptionMergerDirectJoin(unittest.TestCase):
    """总字符 ≤ max_direct_chars 时直接拼接，不调用 LLM。"""

    def test_short_descriptions_no_llm(self):
        merger = DescriptionMerger(max_direct_chars=500)
        descs = ["短描述A", "短描述B"]
        # 不应触发 LLM（_llm 保持 None）
        result = merger.merge(descs)
        self.assertIsNone(merger._llm)
        self.assertIn("短描述A", result)
        self.assertIn("短描述B", result)


class TestDescriptionMergerLLMSummarize(unittest.TestCase):
    """总字符 > max_direct_chars 时触发 LLM 摘要。"""

    def test_long_descriptions_trigger_llm(self):
        merger = DescriptionMerger(max_direct_chars=10)  # 极小阈值，必然触发
        descs = ["这是一段较长的描述内容A", "这是一段较长的描述内容B"]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "合并后的摘要"
        mock_llm.invoke.return_value = mock_response

        with patch("src.ingestion.description_merger.get_llm", return_value=mock_llm):
            result = merger.merge(descs)

        self.assertEqual(result, "合并后的摘要")
        mock_llm.invoke.assert_called_once()

    def test_llm_failure_falls_back_to_join(self):
        merger = DescriptionMerger(max_direct_chars=10)
        descs = ["描述内容AAAA", "描述内容BBBB"]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("网络超时")

        with patch("src.ingestion.description_merger.get_llm", return_value=mock_llm):
            result = merger.merge(descs)

        # 回退到直接拼接
        self.assertIn("描述内容AAAA", result)
        self.assertIn("描述内容BBBB", result)

    def test_llm_lazy_init(self):
        """LLM 只在真正需要时才初始化。"""
        merger = DescriptionMerger(max_direct_chars=10000)
        merger.merge(["短A", "短B"])
        self.assertIsNone(merger._llm)


# ── 2. graph_store 合并写入路径 ───────────────────────────────────────────────

def _make_mock_result(dl, legacy_desc=""):
    """构造模拟 Neo4j 查询结果，同时支持 dl 和 legacy_desc 键。"""
    mock_result = MagicMock()
    mock_result.__getitem__ = lambda _, key: dl if key == "dl" else (legacy_desc if key == "legacy_desc" else None)
    return mock_result


class TestGraphStoreUpsertWithMerge(unittest.TestCase):

    def _make_graph_store(self):
        """构造一个 driver 被 mock 的 GraphStore。"""
        with patch("src.storage.graph_store.GraphDatabase"):
            from src.storage.graph_store import GraphStore
            gs = GraphStore()
            gs.driver = MagicMock()
            return gs

    def _make_session(self, single_return):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value.single.return_value = single_return
        return mock_session

    def test_upsert_entity_new_node(self):
        """新实体：description_list 为空，直接写入新描述；返回 True（新建）。"""
        gs = self._make_graph_store()
        merger = DescriptionMerger()
        mock_session = self._make_session(None)
        gs.driver.session.return_value = mock_session

        is_new = gs.upsert_entity_with_merge("e1", "BERT", "Model", "预训练语言模型", merger)

        self.assertTrue(is_new)
        write_call = mock_session.run.call_args_list[1]
        kwargs = write_call.kwargs if write_call.kwargs else write_call[1]
        self.assertEqual(kwargs.get("desc") or write_call[0][1].get("desc"), "预训练语言模型")

    def test_upsert_entity_existing_node_merges(self):
        """已有实体：新描述追加到 description_list，merger 合并后写回；返回 False（更新）。"""
        gs = self._make_graph_store()
        merger = DescriptionMerger(max_direct_chars=2000)
        mock_session = self._make_session(_make_mock_result(["旧描述A"]))
        gs.driver.session.return_value = mock_session

        is_new = gs.upsert_entity_with_merge("e1", "BERT", "Model", "新描述B", merger)

        self.assertFalse(is_new)
        write_call = mock_session.run.call_args_list[1]
        dl = (write_call.kwargs or write_call[0][1]).get("dl")
        self.assertIn("旧描述A", dl)
        self.assertIn("新描述B", dl)

    def test_upsert_entity_deduplicates(self):
        """相同描述不重复追加。"""
        gs = self._make_graph_store()
        merger = DescriptionMerger()
        mock_session = self._make_session(_make_mock_result(["已有描述"]))
        gs.driver.session.return_value = mock_session

        gs.upsert_entity_with_merge("e1", "BERT", "Model", "已有描述", merger)

        write_call = mock_session.run.call_args_list[1]
        dl = (write_call.kwargs or write_call[0][1]).get("dl")
        self.assertEqual(dl.count("已有描述"), 1)

    def test_upsert_entity_legacy_description_preserved(self):
        """旧节点只有 description 字段（无 description_list）时，旧描述不被丢失。"""
        gs = self._make_graph_store()
        merger = DescriptionMerger(max_direct_chars=2000)
        # description_list 为空，但 legacy_desc 有值（Phase 2.2 之前写入的旧节点）
        mock_session = self._make_session(_make_mock_result([], legacy_desc="旧版描述"))
        gs.driver.session.return_value = mock_session

        gs.upsert_entity_with_merge("e1", "BERT", "Model", "新描述", merger)

        write_call = mock_session.run.call_args_list[1]
        dl = (write_call.kwargs or write_call[0][1]).get("dl")
        self.assertIn("旧版描述", dl)
        self.assertIn("新描述", dl)

    def test_upsert_relation_with_merge(self):
        """关系合并写入：新描述追加到 description_list；返回 True（新建）。"""
        gs = self._make_graph_store()
        merger = DescriptionMerger()
        mock_session = self._make_session(None)
        gs.driver.session.return_value = mock_session

        is_new = gs.upsert_relation_with_merge("e1", "e2", "USES", "用于文本分类", merger)

        self.assertTrue(is_new)
        write_call = mock_session.run.call_args_list[1]
        dl = (write_call.kwargs or write_call[0][1]).get("dl")
        self.assertIn("用于文本分类", dl)

    def test_upsert_relation_legacy_description_preserved(self):
        """旧关系只有 description 字段时，旧描述不被丢失。"""
        gs = self._make_graph_store()
        merger = DescriptionMerger(max_direct_chars=2000)
        mock_session = self._make_session(_make_mock_result([], legacy_desc="旧关系描述"))
        gs.driver.session.return_value = mock_session

        gs.upsert_relation_with_merge("e1", "e2", "USES", "新关系描述", merger)

        write_call = mock_session.run.call_args_list[1]
        dl = (write_call.kwargs or write_call[0][1]).get("dl")
        self.assertIn("旧关系描述", dl)
        self.assertIn("新关系描述", dl)


if __name__ == "__main__":
    unittest.main()
