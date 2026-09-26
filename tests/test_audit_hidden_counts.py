"""已发布隐藏计数表反推审计接口 POST /api/v1/audit-hidden-counts 的端到端测试。"""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/audit-hidden-counts"


def payload(table, row_totals, column_totals, **extra) -> dict:
    body = {
        "table": table,
        "row_totals": row_totals,
        "column_totals": column_totals,
    }
    body.update(extra)
    return body


# ---------------------------------------------------------------- 参考实现

def reference_ranges(table, row_totals, column_totals):
    """独立暴力参考：枚举所有隐藏格正整数填法，返回 (可行, 各隐藏格区间)。

    DFS 逐个隐藏格在 [1, min(本行剩余, 本列剩余)] 内取值并即时剪枝，
    最后要求行、列残量全部恰好为 0。仅用于计数取值很小的测试用例。
    """
    rows = len(table)
    cols = len(table[0])
    hidden = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if table[i][j] is None
    ]
    row_remain = list(row_totals)
    col_remain = list(column_totals)
    for i in range(rows):
        for j in range(cols):
            if table[i][j] is not None:
                row_remain[i] -= table[i][j]
                col_remain[j] -= table[i][j]

    minimums = {}
    maximums = {}

    def dfs(index: int) -> bool:
        if index == len(hidden):
            return all(value == 0 for value in row_remain) and all(
                value == 0 for value in col_remain
            )
        i, j = hidden[index]
        feasible_here = False
        for value in range(1, min(row_remain[i], col_remain[j]) + 1):
            row_remain[i] -= value
            col_remain[j] -= value
            if dfs(index + 1):
                feasible_here = True
                previous = minimums.get((i, j))
                minimums[(i, j)] = (
                    value if previous is None else min(previous, value)
                )
                previous = maximums.get((i, j))
                maximums[(i, j)] = (
                    value if previous is None else max(previous, value)
                )
            row_remain[i] += value
            col_remain[j] += value
        return feasible_here

    if any(value < 0 for value in row_remain) or any(
        value < 0 for value in col_remain
    ):
        feasible = False
    else:
        feasible = dfs(0)
    return feasible, hidden, minimums, maximums


def assert_matches_reference(table, row_totals, column_totals):
    feasible, hidden, minimums, maximums = reference_ranges(
        table, row_totals, column_totals
    )
    response = client.post(
        ENDPOINT, json=payload(table, row_totals, column_totals)
    )
    if not feasible:
        assert response.status_code == 422, response.json()
        return

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["total_hidden"] == len(hidden)
    assert body["row_totals"] == row_totals
    assert body["column_totals"] == column_totals
    assert len(body["hidden_cells"]) == len(hidden)

    # 顺序为行优先，且逐格与暴力枚举的精确区间一致。
    for item, (i, j) in zip(body["hidden_cells"], hidden):
        assert item["row"] == i
        assert item["column"] == j
        assert item["min_count"] == minimums[(i, j)]
        assert item["max_count"] == maximums[(i, j)]
        assert item["inferable"] == (minimums[(i, j)] == maximums[(i, j)])
        assert set(item) == {
            "row",
            "column",
            "min_count",
            "max_count",
            "inferable",
        }
    assert set(body) == {
        "hidden_cells",
        "row_totals",
        "column_totals",
        "total_hidden",
    }


# ---------------------------------------------------------------- 基本功能

def test_two_by_two_all_hidden():
    # x00+x01=3、x10+x11=7、x00+x10=4、x01+x11=6，令 t=x00∈[1,2]。
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None], [None, None]], [3, 7], [4, 6]
        ),
    )
    assert response.status_code == 200
    cells = response.json()["hidden_cells"]
    assert [c["min_count"] for c in cells] == [1, 1, 2, 4]
    assert [c["max_count"] for c in cells] == [2, 2, 3, 5]
    assert all(not c["inferable"] for c in cells)
    assert [(c["row"], c["column"]) for c in cells] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_two_by_two_single_public_cell_all_inferable():
    # 只剩 3 个未知格构成无环悬挂边：逐行逐列即可唯一反推。
    response = client.post(
        ENDPOINT,
        json=payload(
            [[1, None], [None, None]], [11, 20], [11, 20]
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_hidden"] == 3
    assert body["hidden_cells"] == [
        {"row": 0, "column": 1, "min_count": 10, "max_count": 10,
         "inferable": True},
        {"row": 1, "column": 0, "min_count": 10, "max_count": 10,
         "inferable": True},
        {"row": 1, "column": 1, "min_count": 10, "max_count": 10,
         "inferable": True},
    ]


def test_all_hidden_all_ones_are_inferable():
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None], [None, None]], [2, 2], [2, 2]
        ),
    )
    assert response.status_code == 200
    assert all(cell["inferable"] for cell in response.json()["hidden_cells"])
    assert all(cell["min_count"] == 1 for cell in response.json()["hidden_cells"])


def test_empty_hidden_table_returns_empty_list():
    # 没有隐藏格且公开格与合计一致：空列表。
    response = client.post(
        ENDPOINT,
        json=payload([[1, 2], [3, 4]], [3, 7], [4, 6]),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hidden_cells"] == []
    assert body["total_hidden"] == 0
    assert body["row_totals"] == [3, 7]
    assert body["column_totals"] == [4, 6]


def test_directly_consumes_suppress_count_table_output():
    suppress = client.post(
        "/api/v1/suppress-count-table",
        json={
            "counts": [[1, 10, 8], [7, 2, 6], [9, 5, 12]],
            "threshold": 5,
        },
    ).json()
    response = client.post(
        ENDPOINT,
        json=payload(
            suppress["table"],
            suppress["row_totals"],
            suppress["column_totals"],
        ),
    )
    assert response.status_code == 200
    cells = response.json()["hidden_cells"]
    # 4 个隐藏格构成一个 4-环，令 t=x00∈[1,7]。
    assert [(c["row"], c["column"]) for c in cells] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [(c["min_count"], c["max_count"]) for c in cells] == [
        (1, 7),
        (4, 10),
        (1, 7),
        (2, 8),
    ]
    assert all(not c["inferable"] for c in cells)


def test_huge_counts_use_exact_integer_arithmetic():
    # 取值很大也不能枚举数值，结果必须精确。
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, 1], [1, None]],
            [10**12, 10**12],
            [10**12, 10**12],
        ),
    )
    assert response.status_code == 200
    cells = response.json()["hidden_cells"]
    assert all(c["inferable"] for c in cells)
    assert all(c["min_count"] == 10**12 - 1 for c in cells)


def test_supports_dimensions_from_2_to_4():
    # 全隐藏、各行列合计等于列数/行数（每格只能取 1）：覆盖四种尺寸。
    for rows in range(2, 5):
        for cols in range(2, 5):
            table = [[None] * cols for _ in range(rows)]
            response = client.post(
                ENDPOINT,
                json=payload(
                    table, [cols] * rows, [rows] * cols
                ),
            )
            assert response.status_code == 200, (rows, cols, response.json())
            body = response.json()
            assert body["total_hidden"] == rows * cols
            assert all(cell["min_count"] == 1 for cell in body["hidden_cells"])
            assert all(cell["max_count"] == 1 for cell in body["hidden_cells"])


# ---------------------------------------------------------------- 422

def test_422_when_no_filling_exists():
    # 列 1 残量 1 却有 2 个隐藏格（每个至少 1），无可行填法；
    # 行/列残量总和相等（均为 5），必须靠流判定而不是总和判定。
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None], [None, 5]], [2, 9], [6, 5]
        ),
    )
    assert response.status_code == 422


def test_422_when_total_sums_differ():
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None], [None, None]], [2, 2], [3, 2]
        ),
    )
    assert response.status_code == 422
    assert "总和" in response.json()["detail"]


def test_422_when_public_sum_exceeds_total():
    response = client.post(
        ENDPOINT,
        json=payload([[5, None], [None, None]], [4, 9], [6, 7]),
    )
    assert response.status_code == 422


def test_422_when_no_hidden_but_totals_wrong():
    response = client.post(
        ENDPOINT,
        json=payload([[1, 2], [3, 4]], [3, 7], [5, 6]),
    )
    assert response.status_code == 422


def test_422_on_empty_table():
    response = client.post(ENDPOINT, json=payload([], [1], [1, 2]))
    assert response.status_code == 422


def test_422_on_table_not_array():
    for bad in ("x", 5, None):
        response = client.post(
            ENDPOINT, json=payload(bad, [1, 1], [1, 1])
        )
        assert response.status_code == 422, bad


def test_422_on_wrong_row_count():
    response = client.post(
        ENDPOINT,
        json=payload([[None, None]], [2], [1, 1]),
    )
    assert response.status_code == 422
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None] for _ in range(5)], [2] * 5, [5, 5]
        ),
    )
    assert response.status_code == 422


def test_422_on_wrong_column_count():
    response = client.post(
        ENDPOINT, json=payload([[None], [None]], [1, 1], [2])
    )
    assert response.status_code == 422
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None] * 5, [None] * 5], [5, 5], [2] * 5
        ),
    )
    assert response.status_code == 422


def test_422_on_ragged_table():
    response = client.post(
        ENDPOINT,
        json=payload([[None, 2, 3], [4, 5, None]], [5, 9], [4, 7, 3]),
    )
    assert response.status_code == 422


def test_422_on_non_positive_public_cell():
    for bad in (0, -1):
        response = client.post(
            ENDPOINT,
            json=payload(
                [[bad, None], [None, None]], [3, 3], [2, 4]
            ),
        )
        assert response.status_code == 422, bad


def test_422_on_invalid_cell_type():
    for bad in (True, False, 1.5, "3", [1]):
        response = client.post(
            ENDPOINT,
            json=payload(
                [[bad, None], [None, None]], [3, 3], [2, 4]
            ),
        )
        assert response.status_code == 422, bad


def test_422_on_invalid_total():
    base_table = [[None, None], [None, None]]
    for bad in (0, -1, True, False, 2.0, "3", None):
        response = client.post(
            ENDPOINT, json=payload(base_table, [bad, 4], [2, 2])
        )
        assert response.status_code == 422, ("row", bad)
        response = client.post(
            ENDPOINT, json=payload(base_table, [2, 2], [bad, 4])
        )
        assert response.status_code == 422, ("col", bad)


def test_422_on_totals_not_array():
    base_table = [[None, None], [None, None]]
    response = client.post(
        ENDPOINT, json=payload(base_table, 4, [4])
    )
    assert response.status_code == 422
    response = client.post(
        ENDPOINT, json=payload(base_table, [4], 4)
    )
    assert response.status_code == 422


def test_422_on_totals_length_mismatch():
    base_table = [[None, None], [None, None]]
    response = client.post(
        ENDPOINT, json=payload(base_table, [4], [2, 2])
    )
    assert response.status_code == 422
    response = client.post(
        ENDPOINT, json=payload(base_table, [2, 2], [4])
    )
    assert response.status_code == 422


def test_422_on_missing_field():
    assert client.post(
        ENDPOINT, json={"row_totals": [1, 1], "column_totals": [1, 1]}
    ).status_code == 422
    assert client.post(
        ENDPOINT,
        json={"table": [[None, None], [None, None]], "column_totals": [1, 1]},
    ).status_code == 422
    assert client.post(
        ENDPOINT,
        json={"table": [[None, None], [None, None]], "row_totals": [1, 1]},
    ).status_code == 422


def test_422_on_extra_field():
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None], [None, None]],
            [2, 2],
            [2, 2],
            unexpected=1,
        ),
    )
    assert response.status_code == 422


def test_422_on_invalid_json_body():
    response = client.post(
        ENDPOINT, content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_422_detail_mentions_dimension():
    response = client.post(
        ENDPOINT, json=payload([[None, None]], [2], [1, 1])
    )
    assert response.status_code == 422
    assert "2～4 行" in response.json()["detail"]


# ---------------------------------------------------------------- 与暴力枚举比对

def test_matches_brute_force_on_fixed_cases():
    feasible_cases = [
        ([[None, None], [None, None]], [3, 7], [4, 6]),
        ([[1, None], [None, None]], [11, 20], [11, 20]),
        ([[None, None], [None, None]], [2, 2], [2, 2]),
        # suppress-count-table 示例的发布表。
        (
            [[None, None, 8], [None, None, 6], [9, 5, 12]],
            [19, 15, 26],
            [17, 17, 26],
        ),
        # 每行/列仅一个隐藏格：全部可反推。
        ([[None, 2, 3], [4, None, 6], [7, 8, None]], [6, 15, 24], [11, 17, 17]),
        # 3×2 全隐藏。
        ([[None, None], [None, None], [None, None]], [3, 4, 5], [6, 6]),
        # 公开格与隐藏格交错。
        ([[1, None, 2], [None, 3, None]], [6, 8], [3, 6, 5]),
        # 4×4 单个隐藏格：唯一反推。
        (
            [
                [1, 2, 3, None],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            [10, 26, 42, 58],
            [28, 32, 36, 40],
        ),
    ]
    for table, row_totals, column_totals in feasible_cases:
        assert_matches_reference(table, row_totals, column_totals)

    infeasible_cases = [
        ([[None, None], [None, 5]], [2, 9], [6, 5]),
        # 2×2 全隐藏、总和相等但正整数解为空（t 既要 <=1 又要 >=2）。
        ([[None, None], [None, None]], [2, 2], [3, 1]),
        ([[1, 2], [3, 4]], [3, 7], [5, 5]),
        # 行/列残量总和相等，但行 1 只有 1 个隐藏格却要凑出 4。
        ([[None, 1], [1, None]], [1, 5], [2, 4]),
    ]
    for table, row_totals, column_totals in infeasible_cases:
        assert_matches_reference(table, row_totals, column_totals)


def test_matches_brute_force_on_random_cases():
    rng = random.Random(20260926)

    for _ in range(60):
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        # 原始正整数矩阵，保持计数很小以便暴力枚举。
        original = [
            [rng.randint(1, 3) for _ in range(cols)] for _ in range(rows)
        ]
        row_totals = [sum(row) for row in original]
        column_totals = [
            sum(original[i][j] for i in range(rows)) for j in range(cols)
        ]

        all_cells = [(i, j) for i in range(rows) for j in range(cols)]
        hidden_count = rng.randint(0, rows * cols)
        chosen = set(rng.sample(all_cells, hidden_count))
        table = [
            [
                None if (i, j) in chosen else original[i][j]
                for j in range(cols)
            ]
            for i in range(rows)
        ]
        assert_matches_reference(table, row_totals, column_totals)

        # 约一半用例扰动某条合计制造不可行（暴力与接口必须都判不可行）。
        if rng.random() < 0.5:
            perturbed_rows = list(row_totals)
            perturbed_cols = list(column_totals)
            if rng.random() < 0.5:
                perturbed_rows[0] += 1
            else:
                perturbed_cols[0] += 1
            assert_matches_reference(
                table, perturbed_rows, perturbed_cols
            )


# ---------------------------------------------------------------- 示例与既有服务

def test_examples_file_matches_response():
    import json
    from pathlib import Path

    example = json.loads(
        Path("examples/request-audit-hidden-counts.json").read_text()
    )
    response = client.post(ENDPOINT, json=example)
    assert response.status_code == 200
    body = response.json()
    assert body["total_hidden"] == 4
    assert body["hidden_cells"][0] == {
        "row": 0,
        "column": 0,
        "min_count": 1,
        "max_count": 7,
        "inferable": False,
    }


def test_existing_endpoints_still_available():
    assert client.get("/api/v1/health").status_code == 200
    root = client.get("/").json()
    assert root["audit_hidden_counts"] == "/api/v1/audit-hidden-counts"
    assert root["suppress_count_table"] == "/api/v1/suppress-count-table"


def test_no_counts_written_to_disk(tmp_path, monkeypatch):
    """处理请求前后工作目录下不应出现任何数据文件（服务纯内存）。"""
    monkeypatch.chdir(tmp_path)
    response = client.post(
        ENDPOINT,
        json=payload(
            [[None, None], [None, None]], [3, 7], [4, 6]
        ),
    )
    assert response.status_code == 200
    assert list(tmp_path.rglob("*")) == []
