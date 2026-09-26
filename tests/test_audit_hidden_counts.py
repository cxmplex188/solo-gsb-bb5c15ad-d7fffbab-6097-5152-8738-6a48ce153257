"""隐藏格取值区间审计接口 POST /api/v1/audit-hidden-counts 的端到端测试。"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/audit-hidden-counts"


def payload(
    table=None,
    row_totals=None,
    column_totals=None,
    **extra,
) -> dict:
    body = {
        "table": (
            [[None, None, 8], [None, None, 6], [9, 5, 12]]
            if table is None
            else table
        ),
        "row_totals": [19, 15, 26] if row_totals is None else row_totals,
        "column_totals": [17, 17, 26]
        if column_totals is None
        else column_totals,
    }
    body.update(extra)
    return body


# ---------------------------------------------------------------- 参考实现

def reference_audit(table, row_totals, column_totals):
    """独立暴力参考：逐行回溯枚举所有正整数填表方案，记录每隐藏格上下界。

    返回 None 表示无可行方案；否则返回 {(row, col): (min, max)}。
    仅用于小值用例。
    """
    rows = len(table)
    cols = len(table[0])
    hidden = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if table[i][j] is None
    ]
    col_public = [
        sum(table[i][j] for i in range(rows) if table[i][j] is not None)
        for j in range(cols)
    ]

    bounds = {h: [None, None] for h in hidden}
    feasible = False
    assignment: dict[tuple[int, int], int] = {}

    def record() -> None:
        nonlocal feasible
        feasible = True
        for h, value in assignment.items():
            lo, hi = bounds[h]
            bounds[h] = [
                value if lo is None else min(lo, value),
                value if hi is None else max(hi, value),
            ]

    def fill_rows(i: int, col_sum: list[int]) -> None:
        if i == rows:
            if all(col_sum[j] + col_public[j] == column_totals[j] for j in range(cols)):
                record()
            return
        cols_hidden = [j for j in range(cols) if table[i][j] is None]
        remaining = row_totals[i] - sum(
            v for v in table[i] if v is not None
        )

        def fill_row(k: int, left: int) -> None:
            if k == len(cols_hidden):
                if left == 0:
                    fill_rows(i + 1, col_sum)
                return
            j = cols_hidden[k]
            other_hidden = len(cols_hidden) - 1 - k
            rows_below = sum(
                1 for x in range(i + 1, rows) if table[x][j] is None
            )
            for value in range(1, left - other_hidden + 1):
                total = col_sum[j] + value + col_public[j]
                if total > column_totals[j]:
                    break
                # 该列剩余行的隐藏格每行至少 1。
                if total + rows_below > column_totals[j]:
                    break
                assignment[(i, j)] = value
                col_sum[j] += value
                fill_row(k + 1, left - value)
                col_sum[j] -= value
                del assignment[(i, j)]

        fill_row(0, remaining)

    fill_rows(0, [0] * cols)
    if not feasible:
        return None
    return {h: tuple(bounds[h]) for h in hidden}


def assert_matches_reference(body: dict, table, row_totals, column_totals):
    rows = len(table)
    cols = len(table[0])
    ref = reference_audit(table, row_totals, column_totals)
    assert ref is not None
    hidden_cells = body["hidden_cells"]
    # 行优先排序。
    assert [(c["row"], c["column"]) for c in hidden_cells] == sorted(ref)
    for cell in hidden_cells:
        h = (cell["row"], cell["column"])
        assert cell["min_value"] == ref[h][0]
        assert cell["max_value"] == ref[h][1]
        assert cell["inferable"] == (ref[h][0] == ref[h][1])
        assert set(cell) == {
            "row",
            "column",
            "min_value",
            "max_value",
            "inferable",
        }
    assert body["row_totals"] == row_totals
    assert body["column_totals"] == column_totals
    assert body["rows"] == rows
    assert body["columns"] == cols
    assert set(body) == {
        "hidden_cells",
        "row_totals",
        "column_totals",
        "rows",
        "columns",
    }


# ---------------------------------------------------------------- 基本功能

def test_basic_suppress_output():
    # /suppress-count-table 示例的发布表：4 个隐藏格构成一个 4-环，
    # 每格区间由环上自由度决定。
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()
    assert [(c["row"], c["column"]) for c in body["hidden_cells"]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    by = {(c["row"], c["column"]): c for c in body["hidden_cells"]}
    # 自由度为 1 的 4-环：a+b=11、c+d=9、a+c=8、b+d=12。
    assert (by[(0, 0)]["min_value"], by[(0, 0)]["max_value"]) == (1, 7)
    assert (by[(0, 1)]["min_value"], by[(0, 1)]["max_value"]) == (4, 10)
    assert (by[(1, 0)]["min_value"], by[(1, 0)]["max_value"]) == (1, 7)
    assert (by[(1, 1)]["min_value"], by[(1, 1)]["max_value"]) == (2, 8)
    assert not any(c["inferable"] for c in body["hidden_cells"])


def test_single_hidden_cell_is_inferable():
    # 2×2 中只有 1 个隐藏格：行/列合计直接确定其取值 1。
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, 10], [10, 10]],
            row_totals=[11, 20],
            column_totals=[11, 20],
        ),
    )
    assert response.status_code == 200
    cells = response.json()["hidden_cells"]
    assert cells == [
        {
            "row": 0,
            "column": 0,
            "min_value": 1,
            "max_value": 1,
            "inferable": True,
        }
    ]


def test_bridge_hidden_cells_are_inferable():
    # L 形三个隐藏格：(0,1) 由第 1 列合计确定为 1，随后 (0,0)、(1,1)
    # 也被逐行逐列相减唯一确定（悬挂桥边均可反推）。
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, None], [5, None]],
            row_totals=[6, 6],
            column_totals=[6, 6],
        ),
    )
    assert response.status_code == 200
    by = {
        (c["row"], c["column"]): c
        for c in response.json()["hidden_cells"]
    }
    # 第 1 列公开格 (1,0)=5，故 (0,0)=6-5=1；进而 (0,1)=6-1=5、
    # (1,1)=6-5=1：三个悬挂格全部被唯一反推。
    assert by[(0, 0)]["min_value"] == by[(0, 0)]["max_value"] == 1
    assert by[(0, 1)]["min_value"] == by[(0, 1)]["max_value"] == 5
    assert by[(1, 1)]["min_value"] == by[(1, 1)]["max_value"] == 1
    assert all(c["inferable"] for c in by.values())


def test_two_by_two_all_hidden_ranges():
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, None], [None, None]],
            row_totals=[3, 7],
            column_totals=[4, 6],
        ),
    )
    assert response.status_code == 200
    by = {
        (c["row"], c["column"]): c
        for c in response.json()["hidden_cells"]
    }
    # 设 a=(0,0)，则 (0,1)=3-a、(1,0)=4-a、(1,1)=3+a，a∈[1,2]。
    assert (by[(0, 0)]["min_value"], by[(0, 0)]["max_value"]) == (1, 2)
    assert (by[(0, 1)]["min_value"], by[(0, 1)]["max_value"]) == (1, 2)
    assert (by[(1, 0)]["min_value"], by[(1, 0)]["max_value"]) == (2, 3)
    assert (by[(1, 1)]["min_value"], by[(1, 1)]["max_value"]) == (4, 5)
    assert not any(c["inferable"] for c in by.values())


def test_no_hidden_table_returns_empty_list():
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[5, 6], [7, 8]],
            row_totals=[11, 15],
            column_totals=[12, 14],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hidden_cells"] == []
    assert body["rows"] == 2
    assert body["columns"] == 2


def test_dimensions_2_to_4():
    for rows in range(2, 5):
        for cols in range(2, 5):
            matrix = [
                [(i * cols + j) % 5 + 1 for j in range(cols)]
                for i in range(rows)
            ]
            table = [
                [
                    None if (i + j) % 2 == 0 else matrix[i][j]
                    for j in range(cols)
                ]
                for i in range(rows)
            ]
            row_totals = [sum(row) for row in matrix]
            column_totals = [
                sum(matrix[i][j] for i in range(rows)) for j in range(cols)
            ]
            response = client.post(
                ENDPOINT,
                json=payload(
                    table=table,
                    row_totals=row_totals,
                    column_totals=column_totals,
                ),
            )
            assert response.status_code == 200, (rows, cols, response.json())
            assert_matches_reference(
                response.json(), table, row_totals, column_totals
            )


def test_matches_brute_force_on_many_cases():
    # 小值矩阵 + 各种隐藏分布，与逐格回溯暴力枚举比对。
    cases = [
        ([[None, None], [None, None]], [3, 7], [4, 6]),
        ([[None, 10], [10, None]], [12, 12], [12, 12]),
        ([[None, None], [5, None]], [6, 6], [6, 6]),
        ([[None, None, 8], [None, None, 6], [9, 5, 12]], [19, 15, 26], [17, 17, 26]),
        ([[None, 2, 3], [4, None, 6], [7, 8, None]], [6, 15, 24], [12, 15, 18]),
        (
            [[None, None, None, 1], [1, None, None, None], [1, 1, 1, 1]],
            [10, 10, 4],
            [3, 6, 8, 7],
        ),
    ]
    for table, row_totals, column_totals in cases:
        response = client.post(
            ENDPOINT,
            json=payload(
                table=table,
                row_totals=row_totals,
                column_totals=column_totals,
            ),
        )
        assert response.status_code == 200, (table, response.json())
        assert_matches_reference(
            response.json(), table, row_totals, column_totals
        )


def test_large_values_exact_bounds():
    # 大整数 2×2 4-环（无公开格），行合计 [B+7, B+9]、列合计 [B+9, B+7]：
    # 设 a=(0,0)，则 (0,1)=B+7-a、(1,0)=B+9-a、(1,1)=a-2，
    # 由四者均为正整数得 a∈[3, B+6]；验证精确整数算术（不使用浮点）。
    big = 10**12
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, None], [None, None]],
            row_totals=[big + 7, big + 9],
            column_totals=[big + 9, big + 7],
        ),
    )
    assert response.status_code == 200
    by = {
        (c["row"], c["column"]): c
        for c in response.json()["hidden_cells"]
    }
    assert (by[(0, 0)]["min_value"], by[(0, 0)]["max_value"]) == (1, big + 6)
    assert (by[(0, 1)]["min_value"], by[(0, 1)]["max_value"]) == (1, big + 6)
    assert (by[(1, 0)]["min_value"], by[(1, 0)]["max_value"]) == (3, big + 8)
    assert (by[(1, 1)]["min_value"], by[(1, 1)]["max_value"]) == (1, big + 6)
    assert all(not c["inferable"] for c in by.values())


def test_directly_uses_suppress_output():
    # 先调 /suppress-count-table，再把其输出字段直接喂给本接口。
    suppress = client.post(
        "/api/v1/suppress-count-table",
        json={
            "counts": [[1, 10, 8], [7, 2, 6], [9, 5, 12]],
            "threshold": 5,
        },
    ).json()
    response = client.post(
        ENDPOINT,
        json={
            "table": suppress["table"],
            "row_totals": suppress["row_totals"],
            "column_totals": suppress["column_totals"],
        },
    )
    assert response.status_code == 200
    assert_matches_reference(
        response.json(),
        suppress["table"],
        suppress["row_totals"],
        suppress["column_totals"],
    )


def test_examples_file_matches_response():
    example = json.loads(
        Path("examples/request-audit-hidden-counts.json").read_text()
    )
    response = client.post(ENDPOINT, json=example)
    assert response.status_code == 200
    body = response.json()
    assert [(c["row"], c["column"]) for c in body["hidden_cells"]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert all(not c["inferable"] for c in body["hidden_cells"])


# ---------------------------------------------------------------- 422：可行域

def test_422_when_no_feasible_fill_row_too_small():
    # 第 1 行公开格之和 9 > 行合计 5。
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, 9], [1, 1]],
            row_totals=[5, 2],
            column_totals=[2, 5],
        ),
    )
    assert response.status_code == 422
    assert "不存在可行" in response.json()["detail"]


def test_422_when_no_feasible_fill_structure():
    # 公开格与各单侧合计本身不矛盾（行/列余额都为正且够分），但行列
    # 约束无法同时满足：第 1 列合计迫使 (0,0)=6，而第 1 行余额只有 4。
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, None], [5, None]],
            row_totals=[4, 16],
            column_totals=[11, 9],
        ),
    )
    assert response.status_code == 422
    assert "不存在可行" in response.json()["detail"]


def test_422_when_row_and_column_total_sums_differ():
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, None], [None, None]],
            row_totals=[3, 7],
            column_totals=[4, 7],
        ),
    )
    assert response.status_code == 422
    assert "总和" in response.json()["detail"]


def test_422_when_no_hidden_but_totals_mismatch():
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[5, 6], [7, 8]],
            row_totals=[12, 15],
            column_totals=[12, 15],
        ),
    )
    assert response.status_code == 422
    assert "不存在可行" in response.json()["detail"]


def test_422_when_hidden_must_be_positive_but_cannot():
    # 行余额为 0 却有隐藏格：隐藏格至少为 1，不可行。
    response = client.post(
        ENDPOINT,
        json=payload(
            table=[[None, 5], [1, 1]],
            row_totals=[5, 2],
            column_totals=[1, 6],
        ),
    )
    assert response.status_code == 422


# ---------------------------------------------------------------- 422：table

def test_422_on_empty_table():
    assert client.post(
        ENDPOINT, json=payload(table=[])
    ).status_code == 422


def test_422_on_table_not_array():
    for bad in ("x", 5, None, {"a": 1}):
        assert client.post(
            ENDPOINT, json=payload(table=bad)
        ).status_code == 422, bad


def test_422_on_wrong_dimensions():
    assert client.post(
        ENDPOINT, json=payload(table=[[1, 1]])
    ).status_code == 422
    assert client.post(
        ENDPOINT, json=payload(table=[[1, 1] for _ in range(5)])
    ).status_code == 422
    assert client.post(
        ENDPOINT, json=payload(table=[[1], [1]])
    ).status_code == 422
    assert client.post(
        ENDPOINT, json=payload(table=[[1] * 5 for _ in range(2)])
    ).status_code == 422


def test_422_on_ragged_table():
    assert client.post(
        ENDPOINT, json=payload(table=[[None, 1, 2], [1, 2]])
    ).status_code == 422


def test_422_on_row_not_array():
    assert client.post(
        ENDPOINT, json=payload(table=[[1, 2], 3])
    ).status_code == 422


def test_422_on_non_positive_public_cell():
    for bad in (0, -1):
        assert client.post(
            ENDPOINT,
            json=payload(
                table=[[bad, None], [1, 1]],
                row_totals=[4, 2],
                column_totals=[2, 4],
            ),
        ).status_code == 422, bad


def test_422_on_non_integer_public_cell():
    for bad in (True, False, 1.5, "3", [1], {"x": 1}):
        assert client.post(
            ENDPOINT,
            json=payload(
                table=[[bad, None], [1, 1]],
                row_totals=[4, 2],
                column_totals=[2, 4],
            ),
        ).status_code == 422, bad


# ---------------------------------------------------------------- 422：合计

def test_422_on_totals_wrong_length():
    assert client.post(
        ENDPOINT, json=payload(row_totals=[19, 15])
    ).status_code == 422
    assert client.post(
        ENDPOINT, json=payload(column_totals=[17, 17])
    ).status_code == 422


def test_422_on_totals_not_array():
    for bad in (5, "x", None):
        assert client.post(
            ENDPOINT, json=payload(row_totals=bad)
        ).status_code == 422, bad


def test_422_on_non_positive_total():
    for bad in (0, -3):
        assert client.post(
            ENDPOINT, json=payload(row_totals=[bad, 15, 26])
        ).status_code == 422, bad
        assert client.post(
            ENDPOINT, json=payload(column_totals=[bad, 17, 26])
        ).status_code == 422, bad


def test_422_on_non_integer_total():
    for bad in (True, False, 2.5, "5", None):
        assert client.post(
            ENDPOINT, json=payload(row_totals=[bad, 15, 26])
        ).status_code == 422, bad


# ---------------------------------------------------------------- 422：请求结构

def test_422_on_missing_field():
    assert client.post(
        ENDPOINT,
        json={"row_totals": [1], "column_totals": [1]},
    ).status_code == 422
    assert client.post(
        ENDPOINT, json={"table": [[None, 1], [1, 1]], "column_totals": [1, 1]}
    ).status_code == 422
    assert client.post(
        ENDPOINT, json={"table": [[None, 1], [1, 1]], "row_totals": [1, 1]}
    ).status_code == 422


def test_422_on_extra_field():
    assert client.post(
        ENDPOINT, json=payload(unexpected=1)
    ).status_code == 422


def test_422_on_invalid_json_body():
    response = client.post(
        ENDPOINT, content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_422_detail_mentions_dimension_limits():
    response = client.post(ENDPOINT, json=payload(table=[[1, 1]]))
    assert response.status_code == 422
    assert "2～4 行" in response.json()["detail"]


# ---------------------------------------------------------------- 既有服务不受影响

def test_existing_endpoints_still_available():
    assert client.get("/api/v1/health").status_code == 200
    root = client.get("/").json()
    assert root["audit_hidden_counts"] == "/api/v1/audit-hidden-counts"
    assert root["suppress_count_table"] == "/api/v1/suppress-count-table"


def test_no_data_written_to_disk(tmp_path, monkeypatch):
    """处理请求前后工作目录下不应出现任何数据文件（服务纯内存）。"""
    monkeypatch.chdir(tmp_path)
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    assert list(tmp_path.rglob("*")) == []
