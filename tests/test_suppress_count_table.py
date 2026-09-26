"""观测计数表单元格隐藏接口 POST /api/v1/suppress-count-table 的端到端测试。"""

from __future__ import annotations

from itertools import combinations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/suppress-count-table"


_DEFAULT_COUNTS = [[1, 10, 8], [7, 2, 6], [9, 5, 12]]


def payload(counts=_DEFAULT_COUNTS, threshold: int = 5, **extra) -> dict:
    body = {"counts": counts, "threshold": threshold}
    body.update(extra)
    return body


# ---------------------------------------------------------------- 参考实现

def reference_solve(counts: list[list[int]], threshold: int):
    """独立的暴力参考：按补充格数量递增、行优先字典序枚举首个可行方案。"""
    rows = len(counts)
    cols = len(counts[0])
    required = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] < threshold
    ]
    candidates = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if counts[i][j] >= threshold
    ]

    def edge_on_cycle(edge, edges):
        # 删去该边（两个方向都不走）后其行端与列端仍连通即属于环。
        i, j = edge
        target = rows + j
        adjacency: dict[int, list[int]] = {}
        for r, c in edges:
            adjacency.setdefault(r, []).append(rows + c)
            adjacency.setdefault(rows + c, []).append(r)
        stack = [i]
        seen = {i}
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, []):
                if {node, neighbor} == {i, rows + j}:
                    continue
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return target in seen

    if not required:
        return required, []
    for size in range(len(candidates) + 1):
        for chosen in combinations(candidates, size):
            edges = set(required) | set(chosen)
            if all(edge_on_cycle(edge, edges) for edge in required):
                return required, list(chosen)
    raise AssertionError("参考实现也未找到可行方案")


def assert_body_matches_reference(body: dict, counts, threshold):
    rows = len(counts)
    cols = len(counts[0])
    required, secondary = reference_solve(counts, threshold)
    assert body["required_suppressed"] == [[i, j] for i, j in required]
    assert body["secondary_suppressed"] == [[i, j] for i, j in secondary]
    assert body["threshold"] == threshold
    assert body["row_totals"] == [sum(row) for row in counts]
    assert body["column_totals"] == [
        sum(counts[i][j] for i in range(rows)) for j in range(cols)
    ]
    hidden = set(required) | set(secondary)
    expected_table = [
        [None if (i, j) in hidden else counts[i][j] for j in range(cols)]
        for i in range(rows)
    ]
    assert body["table"] == expected_table
    assert set(body) == {
        "table",
        "row_totals",
        "column_totals",
        "required_suppressed",
        "secondary_suppressed",
        "threshold",
    }


# ---------------------------------------------------------------- 基本功能

def test_basic_suppression():
    # 必须隐藏 (0,0)=1 与 (1,1)=2，两者不在同一行/列：最小补充为
    # 行优先字典序最小的一组对角配对 (0,1)=10、(1,0)=7，构成 4-环。
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == [[0, 0], [1, 1]]
    assert body["secondary_suppressed"] == [[0, 1], [1, 0]]
    assert body["table"] == [
        [None, None, 8],
        [None, None, 6],
        [9, 5, 12],
    ]
    assert body["row_totals"] == [19, 15, 26]
    assert body["column_totals"] == [17, 17, 26]
    assert body["threshold"] == 5


def test_two_by_two_single_required_cell():
    # 2×2 中只有 1 个必须隐藏格：其余 3 格全部补充，构成唯一的 4-环。
    response = client.post(
        ENDPOINT, json=payload(counts=[[1, 10], [10, 10]], threshold=5)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == [[0, 0]]
    assert body["secondary_suppressed"] == [[0, 1], [1, 0], [1, 1]]
    assert body["table"] == [[None, None], [None, None]]
    assert body["row_totals"] == [11, 20]
    assert body["column_totals"] == [11, 20]


def test_2x2_all_required_cells():
    # 4 格全部低于阈值：4 格本身构成 4-环，无需补充。
    response = client.post(
        ENDPOINT, json=payload(counts=[[1, 2], [3, 4]], threshold=5)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert body["secondary_suppressed"] == []
    assert body["table"] == [[None, None], [None, None]]
    assert body["row_totals"] == [3, 7]
    assert body["column_totals"] == [4, 6]


def test_no_required_cell_needs_no_suppression():
    response = client.post(
        ENDPOINT, json=payload(counts=[[5, 6], [7, 8]], threshold=5)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == []
    assert body["secondary_suppressed"] == []
    assert body["table"] == [[5, 6], [7, 8]]


def test_threshold_one_hides_nothing():
    # 正整数矩阵下阈值 1：没有严格低于 1 的格。
    response = client.post(
        ENDPOINT, json=payload(counts=[[1, 1], [1, 1]], threshold=1)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == []
    assert body["secondary_suppressed"] == []
    assert body["table"] == [[1, 1], [1, 1]]


def test_adjacent_required_cells_share_column():
    # 两个必须隐藏格共享第 1 列：各补一格形成两个 4-环；行优先字典序
    # 最小的同数方案是都取第 0 列（(0,0) 与 (1,0)）。
    counts = [[6, 1], [7, 2], [8, 9]]
    response = client.post(ENDPOINT, json=payload(counts=counts, threshold=5))
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == [[0, 1], [1, 1]]
    assert body["secondary_suppressed"] == [[0, 0], [1, 0]]
    assert body["table"] == [
        [None, None],
        [None, None],
        [8, 9],
    ]


def test_4x4_matching_four_required_needs_four_secondary():
    # 对角线上 4 个必须隐藏格（两两不共行列）：至多需要 4 个补充格，
    # 3 个补充无法让 4 个格都成环；行优先最小方案沿第 0、1 列配对。
    counts = [
        [1, 9, 9, 9],
        [9, 2, 9, 9],
        [9, 9, 3, 9],
        [9, 9, 9, 4],
    ]
    response = client.post(ENDPOINT, json=payload(counts=counts, threshold=5))
    assert response.status_code == 200
    body = response.json()
    assert body["required_suppressed"] == [[0, 0], [1, 1], [2, 2], [3, 3]]
    # 行优先最小：(0,0) 配 (0,1)+(1,0)，(2,2) 配 (2,3)+(3,2)。
    assert body["secondary_suppressed"] == [
        [0, 1],
        [1, 0],
        [2, 3],
        [3, 2],
    ]
    assert body["row_totals"] == [28, 29, 30, 31]
    assert body["column_totals"] == [28, 29, 30, 31]


def test_supports_dimensions_from_2_to_4():
    for rows in range(2, 5):
        for cols in range(2, 5):
            counts = [
                [(i * cols + j) % 7 + 1 for j in range(cols)]
                for i in range(rows)
            ]
            response = client.post(
                ENDPOINT, json=payload(counts=counts, threshold=4)
            )
            assert response.status_code == 200, (rows, cols, response.json())
            assert_body_matches_reference(response.json(), counts, 4)


def test_matches_brute_force_on_many_cases():
    # 与独立暴力实现比对：覆盖各种阈值、尺寸与必须隐藏格分布。
    cases = [
        ([[1, 10], [10, 10]], 5),
        ([[1, 2], [10, 10]], 5),
        ([[1, 2], [3, 4]], 5),
        ([[9, 9], [9, 1]], 5),
        ([[1, 10, 8], [7, 2, 6], [9, 5, 12]], 5),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 5),
        ([[1, 1, 9], [1, 9, 9], [9, 9, 9]], 5),
        ([[9, 1, 9, 9], [1, 9, 9, 9], [9, 9, 1, 9], [9, 9, 9, 1]], 5),
        ([[1, 9, 9, 9], [9, 2, 9, 9], [9, 9, 3, 9], [9, 9, 9, 4]], 5),
        ([[1, 9, 9], [9, 1, 9], [9, 9, 1]], 5),
        ([[5, 5], [5, 5]], 5),
        ([[10, 10], [10, 10]], 5),
        ([[1, 1], [1, 1]], 100),
        ([[100, 1, 1, 1], [1, 100, 1, 1]], 50),
        ([[1, 1, 100, 100], [1, 100, 100, 1]], 50),
        ([[10**12, 1], [2, 10**12]], 50),
    ]
    for counts, threshold in cases:
        response = client.post(
            ENDPOINT, json=payload(counts=counts, threshold=threshold)
        )
        assert response.status_code == 200, (counts, response.json())
        assert_body_matches_reference(response.json(), counts, threshold)


def test_examples_file_matches_response():
    import json
    from pathlib import Path

    example = json.loads(
        Path("examples/request-suppress-count-table.json").read_text()
    )
    response = client.post(ENDPOINT, json=example)
    assert response.status_code == 200
    assert response.json()["secondary_suppressed"] == [[0, 1], [1, 0]]


# ---------------------------------------------------------------- 422：counts

def test_422_on_empty_counts():
    assert client.post(ENDPOINT, json=payload(counts=[])).status_code == 422


def test_422_on_counts_not_a_matrix():
    for bad in ("x", 5, None):
        assert client.post(
            ENDPOINT, json=payload(counts=bad)
        ).status_code == 422, bad


def test_422_on_wrong_row_count():
    assert client.post(
        ENDPOINT, json=payload(counts=[[1, 1]])
    ).status_code == 422
    assert client.post(
        ENDPOINT,
        json=payload(counts=[[1, 1] for _ in range(5)]),
    ).status_code == 422


def test_422_on_wrong_column_count():
    assert client.post(
        ENDPOINT, json=payload(counts=[[1], [1]])
    ).status_code == 422
    assert client.post(
        ENDPOINT, json=payload(counts=[[1] * 5, [1] * 5])
    ).status_code == 422


def test_422_on_ragged_matrix():
    assert client.post(
        ENDPOINT, json=payload(counts=[[1, 2, 3], [4, 5]])
    ).status_code == 422


def test_422_on_row_not_array():
    assert client.post(
        ENDPOINT, json=payload(counts=[[1, 2], 3])
    ).status_code == 422


def test_422_on_non_positive_count():
    for bad in (0, -1):
        assert client.post(
            ENDPOINT, json=payload(counts=[[bad, 2], [3, 4]])
        ).status_code == 422, bad


def test_422_on_non_integer_count():
    for bad in (True, False, 1.5, "3", None, [1]):
        assert client.post(
            ENDPOINT, json=payload(counts=[[bad, 2], [3, 4]])
        ).status_code == 422, bad


# ---------------------------------------------------------------- 422：threshold

def test_422_on_non_positive_threshold():
    for bad in (0, -5):
        assert client.post(
            ENDPOINT, json=payload(threshold=bad)
        ).status_code == 422, bad


def test_422_on_non_integer_threshold():
    for bad in (True, False, 2.5, "5", None):
        assert client.post(
            ENDPOINT, json=payload(threshold=bad)
        ).status_code == 422, bad


# ---------------------------------------------------------------- 422：请求结构

def test_422_on_missing_field():
    assert client.post(ENDPOINT, json={"counts": [[1, 2], [3, 4]]}).status_code == 422
    assert client.post(ENDPOINT, json={"threshold": 5}).status_code == 422


def test_422_on_extra_field():
    assert client.post(
        ENDPOINT, json=payload(unexpected=1)
    ).status_code == 422


def test_422_on_invalid_json_body():
    response = client.post(
        ENDPOINT, content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_422_detail_mentions_reason():
    response = client.post(
        ENDPOINT, json=payload(counts=[[1, 1]])
    )
    assert response.status_code == 422
    assert "2～4 行" in response.json()["detail"]


# ---------------------------------------------------------------- 既有服务不受影响

def test_existing_endpoints_still_available():
    assert client.get("/api/v1/health").status_code == 200
    root = client.get("/").json()
    assert root["suppress_count_table"] == "/api/v1/suppress-count-table"
    assert root["round_count_table"] == "/api/v1/round-count-table"


def test_no_counts_written_to_disk(tmp_path, monkeypatch):
    """处理请求前后工作目录下不应出现任何数据文件（服务纯内存）。"""
    monkeypatch.chdir(tmp_path)
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    assert list(tmp_path.rglob("*")) == []
