"""Tests for the evaluation layer (Wave 5): metrics, judge and harness."""

from __future__ import annotations

import pytest

from hgrag.eval.dataset import EvalDataset, EvalQuestion, load_jsonl
from hgrag.eval.judge import (
    HeuristicJudge,
    LLMJudge,
    build_judge,
    is_out_of_kb,
    looks_like_refusal,
)
from hgrag.eval.metrics import (
    aggregate,
    bootstrap_ci,
    chunk_recall_at_k,
    citation_precision,
    citation_recall,
    entity_recall,
    extract_citation_markers,
    gold_fact_accuracy,
    mcnemar_test,
    ndcg_at_k,
    no_citation_rate,
    normalize_text,
    paired_bootstrap_ci,
    path_hit_rate,
    percentile,
    rouge_l,
    temporal_accuracy,
    token_f1,
    tokenize,
)
from hgrag.eval.run_eval import (
    ABLATIONS,
    EvalRow,
    compare_systems,
    run_evaluation,
    summarize_run,
    summarize_system,
)

# --------------------------------------------------------------------------
# lexical metrics
# --------------------------------------------------------------------------


def test_normalize_text_strips_diacritics() -> None:
    assert normalize_text("Trần Hưng Đạo!") == "tran hung dao"
    assert normalize_text("  Đại   Việt  ") == "dai viet"


def test_tokenize_removes_stopwords() -> None:
    tokens = tokenize("Trần Hưng Đạo là người chỉ huy quân")
    assert "trần" not in tokens
    assert "hưng" not in tokens
    assert "đạo" not in tokens
    assert "chỉ" not in tokens


def test_token_f1_perfect_and_partial() -> None:
    assert token_f1("Trận Bạch Đằng năm 1285", "Trận Bạch Đằng năm 1285") == pytest.approx(1.0)
    partial = token_f1("Trận Bạch Đằng năm 1285", "Trận Bạch Đằng năm 938")
    assert 0.0 < partial < 1.0
    assert token_f1("hoàn toàn khác", "Trận Bạch Đằng năm 1285") == 0.0


def test_token_f1_handles_empty_strings() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("abc", "") == 0.0


def test_rouge_l_is_lcs_based() -> None:
    assert rouge_l("trận bạch đằng", "trận bạch đằng") == pytest.approx(1.0)
    assert rouge_l("trận bạch", "bạch trận") < 1.0
    assert rouge_l("", "x") == 0.0


# --------------------------------------------------------------------------
# citation metrics
# --------------------------------------------------------------------------


def test_extract_citation_markers() -> None:
    assert extract_citation_markers("a [E1] b [E12] c [e3]") == ["E1", "E12", "E3"]
    assert extract_citation_markers("no citation") == []


def test_citation_precision_with_supported_subset() -> None:
    assert citation_precision("[E1] [E2]", ["E1", "E2"], ["E1"]) == pytest.approx(0.5)
    assert citation_precision("no marker", ["E1"]) == 0.0


def test_citation_precision_falls_back_to_validity() -> None:
    assert citation_precision("[E1]", ["E1"]) == 1.0
    assert citation_precision("[E9]", ["E1"]) == 0.0


def test_citation_recall_with_chunk_mapping() -> None:
    mapping = {"E1": "ck_a", "E2": "ck_b"}
    assert citation_recall("x [E1] y", ["ck_a"], mapping) == pytest.approx(1.0)
    assert citation_recall("x [E2] y", ["ck_a"], mapping) == 0.0


def test_no_citation_rate() -> None:
    assert no_citation_rate(["[E1]", "no citation"]) == pytest.approx(0.5)
    assert no_citation_rate([]) == 0.0


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------


def test_chunk_recall_at_k() -> None:
    assert chunk_recall_at_k(["ck_a"], ["ck_a", "ck_b"], k=10) == 1.0
    assert chunk_recall_at_k(["ck_a", "ck_c"], ["ck_a"], k=10) == pytest.approx(0.5)
    assert chunk_recall_at_k([], ["ck_a"]) == 0.0


def test_ndcg_at_k_rewards_early_hits() -> None:
    early = ndcg_at_k(["ck_a"], ["ck_a", "ck_b", "ck_c"], k=3)
    late = ndcg_at_k(["ck_a"], ["ck_b", "ck_c", "ck_a"], k=3)
    assert early > late
    assert early == pytest.approx(1.0)


def test_entity_recall_normalises_names() -> None:
    assert entity_recall(["Trần Hưng Đạo"], ["Trần Hưng Đạo"]) == 1.0
    assert entity_recall(["Trần Hưng Đạo"], ["Ngô Quyền"]) == 0.0
    assert entity_recall([], ["x"]) == 0.0


def test_path_hit_rate_matches_subsequence() -> None:
    gold = [["A", "B", "C"]]
    assert path_hit_rate(gold, [["A", "B", "C"]]) == 1.0
    assert path_hit_rate(gold, [["A", "X", "B", "C"]]) == 1.0
    assert path_hit_rate(gold, [["A", "B"]]) == 0.0
    assert path_hit_rate([], [["A"]]) == 0.0


def test_path_hit_rate_is_symmetric() -> None:
    """The direction of the traversed edge must not matter."""
    gold = [["Đế Minh", "Đế Minh sinh ra Đế Nghi"]]
    assert path_hit_rate(gold, [["Đế Minh sinh ra Đế Nghi", "Đế Minh"]]) == 1.0


def test_path_hit_rate_matches_substring() -> None:
    """A retrieved node that *contains* the gold node counts as a hit.

    The VHE gold paths name the arguments and the event node; the event node
    is by construction ``"<agent> <verb> <theme>"``.
    """
    gold = [["sinh ra Đế Nghi", "Đế Minh"]]
    retrieved = [["Đế Minh sinh ra Đế Nghi"]]
    assert path_hit_rate(gold, retrieved) == 1.0


def test_temporal_accuracy() -> None:
    assert temporal_accuracy([1285], ["Trận Bạch Đằng năm 1285"]) == 1.0
    assert temporal_accuracy([1285], ["Trận Bạch Đằng năm 938"]) == 0.0
    assert temporal_accuracy([], ["x"]) == 0.0


def test_gold_fact_accuracy_uses_lexical_containment() -> None:
    triples = ["Trần Hưng Đạo -PARTICIPATED_IN-> Trận Bạch Đằng"]
    assert gold_fact_accuracy(triples, "Trần Hưng Đạo tham gia Trận Bạch Đằng") == 1.0
    assert gold_fact_accuracy(triples, "không liên quan") == 0.0


def test_aggregate_ignores_non_numeric() -> None:
    rows = [{"a": 1.0}, {"a": 3.0}, {"a": ""}]
    assert aggregate(rows, ["a"])["a"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_percentile() -> None:
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 50) == pytest.approx(2.5)
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 4.0
    assert percentile([], 50) == 0.0


def test_bootstrap_ci_brackets_the_mean() -> None:
    data = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    result = bootstrap_ci(data, n_samples=500, seed=7)
    assert result.ci_low <= result.mean <= result.ci_high
    assert 0.0 <= result.ci_low and result.ci_high <= 1.0
    assert result.n_samples == 500


def test_bootstrap_ci_is_deterministic() -> None:
    a = bootstrap_ci([0.1, 0.5, 0.9], n_samples=200, seed=3)
    b = bootstrap_ci([0.1, 0.5, 0.9], n_samples=200, seed=3)
    assert a.to_dict() == b.to_dict()


def test_paired_bootstrap_ci_detects_a_better_system() -> None:
    base = [0.0] * 20
    target = [1.0] * 20
    result = paired_bootstrap_ci(target, base, n_samples=500)
    assert result.mean == pytest.approx(1.0)
    assert result.ci_low > 0.9


def test_paired_bootstrap_ci_requires_same_length() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_mcnemar_detects_significant_difference() -> None:
    a = [True] * 12 + [False] * 3
    b = [False] * 12 + [True] * 3
    result = mcnemar_test(a, b)
    assert result.b == 12
    assert result.c == 3
    assert result.n == 15
    assert result.p_value < 0.05
    assert result.significant


def test_mcnemar_no_difference() -> None:
    a = [True, False, True, False]
    b = [True, False, True, False]
    result = mcnemar_test(a, b)
    assert result.n == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_mcnemar_requires_same_length() -> None:
    with pytest.raises(ValueError):
        mcnemar_test([True], [True, False])


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def test_eval_question_roundtrip() -> None:
    q = EvalQuestion(qid="q1", question="Ai là Đế Minh?", answer="Đế Minh")
    restored = EvalQuestion.from_dict(q.to_dict())
    assert restored.qid == "q1"
    assert restored.answer == "Đế Minh"


def test_eval_dataset_stats(tmp_path) -> None:
    dataset = EvalDataset(
        questions=[
            EvalQuestion(qid="q1", question="a", answer="b", question_type="single_hop"),
            EvalQuestion(qid="q2", question="c", question_type="out_of_kb"),
        ]
    )
    stats = dataset.stats()
    assert stats["total"] == 2
    assert stats["out_of_kb"] == 1
    path = tmp_path / "questions.jsonl"
    dataset.save_jsonl(path)
    assert len(load_jsonl(path)) == 2


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------


def test_heuristic_judge_scores_perfect_match() -> None:
    verdict = HeuristicJudge().grade(
        question="q", answer="Trận Bạch Đằng năm 1285", reference="Trận Bạch Đằng năm 1285"
    )
    assert verdict.score == 2.0
    assert verdict.correct
    assert verdict.accuracy == 1.0


def test_heuristic_judge_scores_wrong_answer() -> None:
    verdict = HeuristicJudge().grade(question="q", answer="xyz abc", reference="Trận Bạch Đằng")
    assert verdict.score == 0.0
    assert not verdict.correct


def test_llm_judge_falls_back_to_heuristic_on_error() -> None:
    class _Broken:
        def generate_json(self, request):  # noqa: D102
            raise RuntimeError("boom")

    verdict = LLMJudge(_Broken()).grade(question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng")
    assert verdict.source == "heuristic"


def test_llm_judge_degraded_flag_is_set_on_fallback() -> None:
    """Khi rơi về heuristic, dòng đó phải được đánh dấu là *degraded*.

    Nếu không, bảng so sánh sẽ trộn điểm token-F1 với điểm chấm LLM và
    kết luận trở nên vô nghĩa.
    """

    class _Broken:
        def generate_json(self, request):  # noqa: D102
            raise RuntimeError("boom")

    verdict = LLMJudge(_Broken(), max_retries=1, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert verdict.degraded is True


def test_llm_judge_retries_before_falling_back() -> None:
    """Một lần gọi hỏng (429 tạm thời) không được làm hỏng cả lượt chấm."""

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("429 tạm thời")
            return {
                "score": 2,
                "correct": True,
                "grounded": True,
                "supported_evidence_ids": ["E1"],
                "reason": "ok",
            }

    client = _Flaky()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 3
    assert verdict.source == "llm"
    assert verdict.degraded is False
    assert verdict.score == 2.0


def test_llm_judge_rejects_a_self_contradictory_verdict() -> None:
    """Verdict tự mâu thuẫn (lý do khen nhưng điểm = 0) phải bị loại.

    Gemma thỉnh thoảng trả về `score=0` kèm lý do "Đáp án đúng về nghĩa và
    đầy đủ...".  Verdict như vậy không thể tin: nó đưa một đáp án đúng vào
    nhóm "sai" của báo cáo.  Judge phải chấm lại, và chỉ khi mọi lần đều
    mâu thuẫn thì mới rơi về heuristic (đánh dấu ``degraded``).
    """

    class _Contradictory:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            return {
                "score": 0,
                "correct": False,
                "grounded": True,
                "supported_evidence_ids": ["E1"],
                "reason": "Đáp án đúng về nghĩa và đầy đủ so với đáp án tham chiếu.",
            }

    client = _Contradictory()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 3
    assert verdict.source == "heuristic"
    assert verdict.degraded is True


def test_llm_judge_accepts_a_critique_containing_a_praise_phrase() -> None:
    """Lý do *chê* chứa cụm khen không được bị loại.

    Đây là hồi quy của lỗi đo được trên cache thật: `"Đáp án ứng viên **không
    trả lời đúng** mốc liên hệ..."` chứa chuỗi `"trả lời đúng"`, nên bộ lọc
    tìm chuỗi cũ coi là mâu thuẫn và loại verdict.  Hậu quả: dòng rơi về
    heuristic token-F1 -- một heuristic *chặt hơn* rubric -- nên một đáp án
    hoàn toàn đúng bị báo cáo là sai.
    """

    class _Critique:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            return {
                "score": 0,
                "correct": False,
                "grounded": True,
                "supported_evidence_ids": [],
                "reason": (
                    "Đáp án ứng viên không trả lời đúng mốc liên hệ giữa Phú Điền "
                    "và Bình theo tài liệu, mà đưa ra thông tin sai."
                ),
            }

    client = _Critique()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 1
    assert verdict.source == "llm"
    assert verdict.degraded is False
    assert verdict.score == 0.0


def test_llm_judge_accepts_a_cancelled_praise_by_a_contrastive_clause() -> None:
    """Lý do có mệnh đề khen bị từ nối đảo chiều lật ngược không mâu thuẫn.

    *"Đáp án sai mốc thời gian 1416 thay vì 1418. Thông tin về Hội thề Lũng
    Nhai là **đúng về nghĩa** nhưng sai chi tiết quan trọng."* -- điểm 0 là
    đúng; mệnh đề khen đã bị `"nhưng"` vô hiệu hoá.
    """

    class _Cancelled:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            return {
                "score": 0,
                "correct": False,
                "grounded": True,
                "supported_evidence_ids": [],
                "reason": (
                    "Đáp án sai mốc thời gian 1416 thay vì 1418. Thông tin về Hội "
                    "thề Lũng Nhai là đúng về nghĩa nhưng sai chi tiết quan trọng."
                ),
            }

    client = _Cancelled()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 1
    assert verdict.source == "llm"
    assert verdict.degraded is False


def test_llm_judge_still_rejects_an_unqualified_praise_with_a_low_score() -> None:
    """Mệnh đề khen *đứng độc lập* vẫn phải bị loại.

    Đây là dạng mâu thuẫn thật: `"Đáp án đúng hoàn toàn so với đáp án tham
    chiếu..."` với `score=0` (đo được trên `vhe_00001`).
    """

    class _Unqualified:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            return {
                "score": 0,
                "correct": False,
                "grounded": True,
                "supported_evidence_ids": ["E1", "E2"],
                "reason": (
                    "Đáp án đúng hoàn toàn so với đáp án tham chiếu và gold facts. "
                    "Thông tin được hỗ trợ bởi E1 và E2."
                ),
            }

    client = _Unqualified()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 3
    assert verdict.source == "heuristic"
    assert verdict.degraded is True


def test_split_clauses_keeps_the_critique_apart_from_the_praise() -> None:
    """Tách mệnh đề phải giữ lại đủ ngữ cảnh để phân biệt khen và chê."""

    from hgrag.eval.judge import _split_clauses

    clauses = _split_clauses(
        "Đáp án sai mốc thời gian 1416 thay vì 1418. Thông tin về Hội thề Lũng "
        "Nhai là đúng về nghĩa nhưng sai chi tiết quan trọng."
    )
    assert any("đúng về nghĩa" in c for c in clauses)
    assert any("sai chi tiết" in c for c in clauses)
    # Mệnh đề khen không được nuốt mất từ nối đảo chiều.
    assert not any("nhưng" in c.lower() for c in clauses)


def test_llm_judge_retry_changes_the_prompt_not_only_the_cache_key() -> None:
    """Lần chấm lại phải **đổi prompt**, không chỉ đổi cache key.

    Với ``temperature=0`` cùng một prompt luôn cho cùng một kết quả, nên
    nếu chỉ đổi ``task`` thì các lần thử trả về đúng những payload giống
    hệt nhau (đo được trên ``vhe_00001``).  Đây là lý do ba lần thử trước đây
    không giải quyết được verdict tự mâu thuẫn.
    """

    class _Recorder:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_json(self, request):  # noqa: D102
            self.prompts.append(request.messages[-1].content)
            # Lần đầu mâu thuẫn (khen nhưng điểm 0), các lần sau hợp lệ.
            if len(self.prompts) == 1:
                return {
                    "score": 0,
                    "correct": False,
                    "grounded": True,
                    "supported_evidence_ids": ["E1"],
                    "reason": "Đáp án đúng hoàn toàn so với đáp án tham chiếu.",
                }
            return {
                "score": 2,
                "correct": True,
                "grounded": True,
                "supported_evidence_ids": ["E1"],
                "reason": "Đáp án đúng và đầy đủ so với đáp án tham chiếu.",
            }

    client = _Recorder()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert len(client.prompts) == 2
    assert client.prompts[0] != client.prompts[1]
    assert verdict.source == "llm"
    assert verdict.degraded is False
    assert verdict.score == 2.0
    assert verdict.correct is True


def test_llm_judge_rejects_a_repetitive_reason() -> None:
    """Lý do lặp vòng là dấu hiệu mô hình hết ngân sách giữa chừng."""

    class _Repetitive:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            return {
                "score": 2,
                "correct": True,
                "grounded": True,
                "supported_evidence_ids": ["E1"],
                "reason": (
                    "Đáp án ứng viên là một câu hoàn chỉnh. "
                    "Đáp án ứng viên là một câu câu hoàn chỉnh. "
                    "Đáp án ứng viên là một câu hoàn chỉnh."
                ),
            }

    client = _Repetitive()
    verdict = LLMJudge(client, max_retries=2, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 2
    assert verdict.source == "heuristic"
    assert verdict.degraded is True


def test_llm_judge_accepts_a_consistent_verdict() -> None:
    """Verdict nhất quán phải được giữ nguyên (không chấm lại)."""

    class _Consistent:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, request):  # noqa: D102
            self.calls += 1
            return {
                "score": 2,
                "correct": True,
                "grounded": True,
                "supported_evidence_ids": ["E1"],
                "reason": "Đáp án đúng về nghĩa và đầy đủ so với đáp án tham chiếu.",
            }

    client = _Consistent()
    verdict = LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="Trận Bạch Đằng", reference="Trận Bạch Đằng"
    )
    assert client.calls == 1
    assert verdict.source == "llm"
    assert verdict.score == 2.0
    assert verdict.degraded is False


def test_llm_judge_vary_the_task_on_retry() -> None:
    """Lần chấm lại phải dùng cache key khác, nếu không cache sẽ trả lại
    verdict hỏng cũ."""

    class _Tracking:
        def __init__(self) -> None:
            self.tasks: list[str] = []

        def generate_json(self, request):  # noqa: D102
            self.tasks.append(request.task)
            return {
                "score": 2,
                "correct": True,
                "grounded": True,
                "supported_evidence_ids": [],
                "reason": "ok",
            }

    client = _Tracking()
    LLMJudge(client, max_retries=3, retry_base_delay=0.0).grade(
        question="q", answer="a", reference="a"
    )
    assert client.tasks == ["judge_answer"]


# --------------------------------------------------------------------------
# report formatting
# --------------------------------------------------------------------------


def test_fmt_signed_makes_the_direction_of_a_difference_explicit() -> None:
    """Bảng Δ phải chỉ rõ dấu.

    Không có dấu thì ``0.080`` trong cột "Δ so với Ours" bị đọc nhầm thành
    "Ours tốt hơn 8 %" trong khi nó có nghĩa là "ablation này **tốt hơn** Ours
    8 %".
    """
    from hgrag.eval.report import _fmt, _fmt_signed

    assert _fmt_signed(0.08) == "+0.080"
    assert _fmt_signed(-0.08) == "-0.080"
    assert _fmt_signed(0.0) == "+0.000"
    assert _fmt_signed(None) == "–"
    # Giá trị không có dấu vẫn giữ nguyên hành vi cũ.
    assert _fmt(0.08) == "0.080"


def test_build_judge_picks_heuristic_without_client() -> None:
    assert isinstance(build_judge(None), HeuristicJudge)
    assert isinstance(build_judge(object(), mock=True), HeuristicJudge)


# --------------------------------------------------------------------------
# out-of-KB behaviour rubric
# --------------------------------------------------------------------------


def test_is_out_of_kb_detects_question_type_and_source() -> None:
    assert is_out_of_kb(EvalQuestion(qid="q1", question="q", question_type="out_of_kb"))
    assert is_out_of_kb(EvalQuestion(qid="q2", question="q", source="out_of_kb"))
    assert not is_out_of_kb(
        EvalQuestion(qid="q3", question="q", question_type="single_hop", source="vhe_template")
    )


def test_looks_like_refusal() -> None:
    assert looks_like_refusal(
        "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."
    )
    assert looks_like_refusal("Câu hỏi này ngoài phạm vi kiến thức.")
    assert not looks_like_refusal("Ngô Quyền đã đánh bại quân Nam Đô năm 938.")


def test_out_of_kb_refusal_is_scored_correct() -> None:
    """A refusal is the *correct* behaviour for an out-of-KB question."""
    verdict = HeuristicJudge().grade(
        question="q",
        answer="Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.",
        reference="",
        out_of_kb=True,
    )
    assert verdict.score == 2.0
    assert verdict.correct
    assert verdict.grounded


def test_out_of_kb_answer_is_scored_as_hallucination() -> None:
    verdict = HeuristicJudge().grade(
        question="q",
        answer="Thủ tướng hiện nay là ông Nguyễn Văn A.",
        reference="",
        out_of_kb=True,
    )
    assert verdict.score == 0.0
    assert not verdict.correct
    assert not verdict.grounded


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class _StubPipeline:
    """A pipeline stub that always answers the same thing."""

    def __init__(self, answer: str = "Đáp án [E1]") -> None:
        self.answer = answer
        self.llm_client = None
        #: The ablation flags that were passed to the last ``ask()`` call.
        self.ablations: list[dict] = []

    def ask(self, question, ablation=None, system="ours", **kwargs):  # noqa: D102
        self.ablations.append(dict(ablation or {}))
        return _StubResult(self.answer)

    def answer_llm_only(self, question, system="B0"):  # noqa: D102
        return _StubResult(self.answer)

    def answer_vanilla_rag(self, question, system="B1"):  # noqa: D102
        return _StubResult(self.answer)

    def answer_rag_plus_graph(self, question, system="B2"):  # noqa: D102
        return _StubResult(self.answer)


class _StubResult:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.citations = ["E1"]
        self.reasoning_chain = []
        self.evidence = [
            {
                "evidence_id": "E1",
                "kind": "chunk",
                "text": "Trận Bạch Đằng năm 1285",
                "score": 1.0,
                "meta": {"chunk_id": "ck_a", "title": "Trần Hưng Đạo"},
            }
        ]
        self.explanation = {
            "linked_entities": [{"name": "Trần Hưng Đạo"}],
            "paths": [{"nodes": [{"name": "Trần Hưng Đạo"}, {"name": "Trận Bạch Đằng"}]}],
            "subgraph": {"nodes": [], "edges": []},
            "retrieval_trace": {"n_vector_hits": 3},
        }
        self.understanding = {}
        self.refused = False
        self.elapsed = 0.5
        self.system = "ours"


def _dataset() -> EvalDataset:
    return EvalDataset(
        questions=[
            EvalQuestion(
                qid="q1",
                question="Trận Bạch Đằng năm nào?",
                answer="Trận Bạch Đằng năm 1285",
                question_type="single_hop",
                gold_entities=["Trần Hưng Đạo"],
                gold_chunk_ids=["ck_a"],
                gold_relations=["Trần Hưng Đạo -PARTICIPATED_IN-> Trận Bạch Đằng"],
                gold_paths=[["Trần Hưng Đạo", "Trận Bạch Đằng"]],
                year_refs=[1285],
            ),
            EvalQuestion(
                qid="q2",
                question="Ai là thủ tướng hiện nay?",
                question_type="out_of_kb",
            ),
        ]
    )


def test_run_evaluation_produces_rows_for_every_system() -> None:
    run = run_evaluation(
        _StubPipeline(), _dataset(), systems=("B0", "B1", "B2", "ours"), ablations=False, judge=HeuristicJudge()
    )
    assert len(run.rows) == 8  # 4 systems x 2 questions
    assert {r.system for r in run.rows} == {"B0", "B1", "B2", "ours"}


def test_run_evaluation_includes_ablation_rows() -> None:
    run = run_evaluation(
        _StubPipeline(), _dataset(), systems=("ours",), ablations=True, judge=HeuristicJudge()
    )
    assert len(ABLATIONS) == 7
    assert {r.system for r in run.rows} == {"ours", *ABLATIONS.keys()}


def test_named_ablation_system_actually_disables_its_component() -> None:
    """A system named after an ablation must disable that component.

    Nếu không, ``A4_no_graph`` chỉ là một bản sao của ``ours`` và bảng
    ablation trong báo cáo vô nghĩa.
    """
    pipeline = _StubPipeline()
    run = run_evaluation(
        pipeline,
        _dataset(),
        systems=("ours", "A4_no_graph"),
        ablations=False,
        judge=HeuristicJudge(),
    )
    assert {r.system for r in run.rows} == {"ours", "A4_no_graph"}
    # `ours` chạy pipeline đầy đủ, còn `A4_no_graph` phải tắt thành phần đồ thị.
    assert pipeline.ablations[-1] == {"no_graph": True}


def test_explicit_ablation_systems_are_not_duplicated() -> None:
    """Chỉ định sẵn hệ thống ablation thì không được chạy lại nó."""
    run = run_evaluation(
        _StubPipeline(),
        _dataset(),
        systems=("ours", "A4_no_graph"),
        ablations=True,
        judge=HeuristicJudge(),
    )
    keys = [(r.system, r.qid) for r in run.rows]
    assert len(keys) == len(set(keys)), "một cặp (hệ thống, câu hỏi) bị chạy trùng"
    # `A4_no_graph` đã có trong --systems nên không được thêm lần nữa.
    assert len({r.system for r in run.rows}) == 2 + len(ABLATIONS) - 1


def test_run_evaluation_scores_retrieval_metrics() -> None:
    run = run_evaluation(
        _StubPipeline(), _dataset(), systems=("ours",), ablations=False, judge=HeuristicJudge()
    )
    row = [r for r in run.rows if r.qid == "q1"][0]
    assert row.chunk_recall_at_10 == 1.0
    assert row.path_hit == 1.0
    assert row.entity_recall == 1.0
    assert row.temporal_accuracy == 1.0
    assert row.citation_precision == 1.0


def test_summarize_system_reports_groups() -> None:
    run = run_evaluation(
        _StubPipeline(), _dataset(), systems=("ours",), ablations=False, judge=HeuristicJudge()
    )
    summary = summarize_system(run, "ours")
    assert summary["n"] == 2
    assert "single_hop" in summary["by_type"]
    assert summary["latency_p50"] >= 0


def test_compare_systems_returns_mcnemar_and_ci() -> None:
    run = run_evaluation(
        _StubPipeline(), _dataset(), systems=("B1", "ours"), ablations=False, judge=HeuristicJudge()
    )
    comparison = compare_systems(run, baseline="B1", target="ours")
    assert comparison["n_questions"] == 2
    assert "accuracy_diff_ci" in comparison
    assert "mcnemar" in comparison


def test_summarize_run_orders_systems() -> None:
    run = run_evaluation(
        _StubPipeline(), _dataset(), systems=("B0", "B1", "B2", "ours"), ablations=False, judge=HeuristicJudge()
    )
    summaries = summarize_run(run)
    assert [s["system"] for s in summaries] == ["B0", "B1", "B2", "ours"]


def test_eval_row_to_dict_is_json_serialisable() -> None:
    import json

    row = EvalRow(qid="q1", system="ours", question="q")
    json.dumps(row.to_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------
# checkpoint / resume
# --------------------------------------------------------------------------


class _CountingPipeline(_StubPipeline):
    """Đếm số lần được gọi (để kiểm tra resume có bỏ qua việc đã làm)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def ask(self, question, ablation=None, system="ours", **kwargs):  # noqa: D102
        self.calls += 1
        return super().ask(question, ablation=ablation, system=system, **kwargs)


def test_checkpoint_is_written_and_reused(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.json"
    pipeline = _CountingPipeline()
    run = run_evaluation(
        pipeline,
        _dataset(),
        systems=("ours",),
        ablations=False,
        judge=HeuristicJudge(),
        checkpoint_path=ckpt,
        checkpoint_every=1,
    )
    assert ckpt.exists()
    assert len(run.rows) == 2
    assert pipeline.calls == 2

    # Chạy lại: pipeline mới, cùng checkpoint -> không gọi lại câu đã chấm.
    pipeline2 = _CountingPipeline()
    run2 = run_evaluation(
        pipeline2,
        _dataset(),
        systems=("ours",),
        ablations=False,
        judge=HeuristicJudge(),
        checkpoint_path=ckpt,
        checkpoint_every=1,
    )
    assert pipeline2.calls == 0
    assert len(run2.rows) == 2


def test_checkpoint_resume_keeps_previous_rows(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.json"
    run_evaluation(
        _CountingPipeline(),
        _dataset(),
        systems=("ours",),
        ablations=False,
        judge=HeuristicJudge(),
        checkpoint_path=ckpt,
        checkpoint_every=1,
    )
    run = run_evaluation(
        _CountingPipeline(),
        _dataset(),
        systems=("ours",),
        ablations=False,
        judge=HeuristicJudge(),
        checkpoint_path=ckpt,
        checkpoint_every=1,
        skip_existing=False,
    )
    assert len(run.rows) == 2


# --------------------------------------------------------------------------
# error analysis
# --------------------------------------------------------------------------


def _row(**kwargs) -> EvalRow:
    base = {
        "qid": "q1",
        "system": "ours",
        "question": "Câu hỏi?",
        "question_type": "single_hop",
        "answer": "Một câu trả lời có dẫn chiếu [E1].",
        "judge_correct": False,
        "judge_score": 0.0,
        "n_evidence": 3,
        "n_citations": 1,
        "citation_precision": 1.0,
        "chunk_recall_at_10": 1.0,
        "entity_recall": 1.0,
        "path_hit": 1.0,
    }
    base.update(kwargs)
    return EvalRow(**base)


def test_classify_error_detects_refusal_on_answerable_question() -> None:
    from hgrag.eval.error_analysis import classify_error

    row = _row(refused=True, answer="Tôi không tìm thấy đủ thông tin trong kho.")
    assert classify_error(row, "single_hop") == "refused_when_answerable"


def test_classify_error_detects_out_of_kb_hallucination() -> None:
    from hgrag.eval.error_analysis import classify_error

    row = _row(answer="Năm 1945, Việt Nam độc lập.", refused=False)
    assert classify_error(row, "out_of_kb") == "out_of_kb_hallucination"


def test_classify_error_blames_the_question_when_the_judge_does() -> None:
    from hgrag.eval.error_analysis import classify_error

    row = _row(
        judge_reason="Câu hỏi ghép các sự kiện đầu thế kỷ XX vào bối cảnh triều Trần là sai lệch lịch sử.",
    )
    assert classify_error(row, "single_hop") == "bad_question"


def test_classify_error_detects_ungrounded_answer() -> None:
    from hgrag.eval.error_analysis import classify_error

    row = _row(judge_reason="Đáp án chứa nhiều thông tin không có trong CONTEXT.")
    assert classify_error(row, "single_hop") == "generation_wrong_despite_evidence"


def test_classify_error_detects_missing_evidence() -> None:
    from hgrag.eval.error_analysis import classify_error

    row = _row(n_evidence=0, answer="Không có gì.", n_citations=0)
    assert classify_error(row, "single_hop") == "missing_evidence"


def test_analyze_errors_counts_categories() -> None:
    from hgrag.eval.error_analysis import analyze_errors
    from hgrag.eval.run_eval import EvalRun

    rows = [
        _row(qid="a", refused=True, answer="Tôi không tìm thấy đủ thông tin trong kho."),
        _row(qid="b", judge_reason="Câu hỏi sai lệch lịch sử."),
    ]
    run = EvalRun(rows=rows, meta={})
    analysis = analyze_errors(run, "ours")
    assert analysis["n_failures"] == 2
    assert analysis["counts"]["refused_when_answerable"] == 1
    assert analysis["counts"]["bad_question"] == 1
