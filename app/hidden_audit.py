"""已发布计数表中隐藏格的精确取值区间审计。

``/suppress-count-table`` 发布的表格把被隐藏的格置为 ``null``，同时公开
行合计与列合计。即便隐藏格已处在隐藏边构成的环上，攻击者通常也只能把
该格的取值缩小到一个区间，而非唯一值。本模块对每个隐藏格，在所有满足
下列条件的**填表方案**中求其精确最小值与最大值：

- 公开格固定为发布的正整数；
- 每个隐藏格为正整数（>= 1）；
- 每行、每列的格值之和精确等于公开的行列合计。

隐藏格是变量，公开格从行列合计中扣除常量，问题化为行-列二部图上的
**带下界整数流**：隐藏边为“行节点 -> 列节点”的边，容量下界 1、上界
为隐藏值总和，行列节点的流量需求为扣除公开格后的余额。二部图关联矩阵
全单模，整数最小费用流给出的取值必为整数，因此结果精确。

对某个隐藏边求最小值/最大值时，只需在所有流量等于总值的可行流中令该边
费用为 +1 / -1、其余边费用为 0：最小费用流即最小化/最大化该边流量。
不存在任何可行方案（含行列两侧总和不一致）时抛 422。

矩阵至多 4×4（隐藏边至多 16），每个隐藏格各跑两次最小费用流，规模
微不足道。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .anonymizer import unprocessable

# 矩阵行数与列数的闭区间限制。
MIN_DIMENSION = 2
MAX_DIMENSION = 4


def prepare_table(raw_table: Any) -> list[list[int | None]]:
    """校验发布表，返回公开格为正整数、隐藏格为 None 的矩形矩阵。

    矩阵必须为 2～4 行、2～4 列的矩形；公开格必须是正整数（拒绝布尔、
    浮点与字符串），隐藏格必须是 null。
    """
    if not isinstance(raw_table, list) or not raw_table:
        raise unprocessable("table 必须是至少包含 2 行的非空数组")
    if not MIN_DIMENSION <= len(raw_table) <= MAX_DIMENSION:
        raise unprocessable(
            f"table 必须包含 {MIN_DIMENSION}～{MAX_DIMENSION} 行，"
            f"当前为 {len(raw_table)} 行"
        )

    matrix: list[list[int | None]] = []
    width: int | None = None
    for index, raw_row in enumerate(raw_table):
        label = f"table 第 {index + 1} 行"
        if not isinstance(raw_row, list):
            raise unprocessable(f"{label}必须是数组")
        if not MIN_DIMENSION <= len(raw_row) <= MAX_DIMENSION:
            raise unprocessable(
                f"{label}必须包含 {MIN_DIMENSION}～{MAX_DIMENSION} 列，"
                f"当前为 {len(raw_row)} 列"
            )
        if width is None:
            width = len(raw_row)
        elif len(raw_row) != width:
            raise unprocessable(
                f"{label}的列数 {len(raw_row)} 与首行的 {width} 列不一致"
            )
        row: list[int | None] = []
        for value in raw_row:
            if value is None:
                row.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise unprocessable(
                    f"{label}的公开格必须是整数，隐藏格必须为 null"
                )
            if value < 1:
                raise unprocessable(f"{label}的公开格必须是正整数（>= 1）")
            row.append(value)
        matrix.append(row)
    return matrix


def prepare_totals(raw_totals: Any, expected_length: int, name: str,
                   dimension: str) -> list[int]:
    """校验一侧公开合计：长度匹配的正整数数组。"""
    if not isinstance(raw_totals, list):
        raise unprocessable(f"{name} 必须是数组")
    if len(raw_totals) != expected_length:
        raise unprocessable(
            f"{name} 的长度 {len(raw_totals)} 必须等于 table 的{dimension}数 "
            f"{expected_length}"
        )

    totals: list[int] = []
    for value in raw_totals:
        if isinstance(value, bool) or not isinstance(value, int):
            raise unprocessable(f"{name} 的每项必须是整数")
        if value < 1:
            raise unprocessable(f"{name} 的每项必须是正整数（>= 1）")
        totals.append(value)
    return totals


def audit_hidden_counts(
    table: list[list[int | None]],
    row_totals: list[int],
    column_totals: list[int],
) -> list[dict[str, Any]]:
    """对每个隐藏格求精确 [最小值, 最大值] 区间，按行优先返回。

    每项为 ``{"row", "column", "min_value", "max_value", "inferable"}``；
    ``inferable`` 在上下界相等（取值被唯一反推）时为 true。无隐藏格时
    返回空列表（公开格必须与行列合计完全吻合，否则 422）。不存在任何
    可行填表方案时抛 422。
    """
    rows = len(table)
    cols = len(table[0])

    # 扣除公开格常量后的行/列余额，即该行/列隐藏格之和。
    row_remaining = [
        row_totals[i] - sum(value for value in table[i] if value is not None)
        for i in range(rows)
    ]
    col_remaining = [
        column_totals[j]
        - sum(table[i][j] for i in range(rows) if table[i][j] is not None)
        for j in range(cols)
    ]
    hidden = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if table[i][j] is None
    ]

    if sum(row_totals) != sum(column_totals):
        raise unprocessable(
            f"row_totals 的总和 {sum(row_totals)} 与 column_totals 的总和 "
            f"{sum(column_totals)} 必须相等，不存在可行填表方案"
        )

    for i, remaining in enumerate(row_remaining):
        if remaining < 0:
            raise unprocessable(
                f"第 {i + 1} 行的公开格之和 {row_totals[i] - remaining} "
                f"已超过公开行合计 {row_totals[i]}，不存在可行填表方案"
            )
    for j, remaining in enumerate(col_remaining):
        if remaining < 0:
            raise unprocessable(
                f"第 {j + 1} 列的公开格之和 {column_totals[j] - remaining} "
                f"已超过公开列合计 {column_totals[j]}，不存在可行填表方案"
            )

    total = sum(row_remaining)
    if sum(col_remaining) != total:
        # 两侧总和相等且公开格总和相同，两侧余额总和必相等；此处仅作防御。
        raise unprocessable("行列余额不一致，不存在可行填表方案")

    if not hidden:
        # 无隐藏格：公开格之和必须与公开行列合计逐行逐列吻合。
        if any(remaining != 0 for remaining in row_remaining) or any(
            remaining != 0 for remaining in col_remaining
        ):
            raise unprocessable(
                "表中没有隐藏格，但公开格之和与公开行列合计不一致，"
                "不存在可行填表方案"
            )
        return []

    # 每条隐藏边下界为 1（隐藏格必须为正整数）：先把下界流量预占，
    # 再在“下界为 0”的剩余流上分配 total - hidden_count。
    degree_rows = [0] * rows
    degree_cols = [0] * cols
    for i, j in hidden:
        degree_rows[i] += 1
        degree_cols[j] += 1
    row_demand = [row_remaining[i] - degree_rows[i] for i in range(rows)]
    col_demand = [col_remaining[j] - degree_cols[j] for j in range(cols)]

    if any(demand < 0 for demand in row_demand):
        raise unprocessable(
            "某行隐藏格数量多于该行可分配的余额（隐藏格均须为正整数），"
            "不存在可行填表方案"
        )
    if any(demand < 0 for demand in col_demand):
        raise unprocessable(
            "某列隐藏格数量多于该列可分配的余额（隐藏格均须为正整数），"
            "不存在可行填表方案"
        )

    residual_total = total - len(hidden)

    # 先做一次可行性判定（费用全 0），不可行时直接 422，避免逐边重复计算。
    feasibility_flow, _ = _min_cost_flow(
        hidden, row_demand, col_demand, residual_total, None, 0
    )
    if feasibility_flow < residual_total:
        raise unprocessable(
            "公开格与行列合计约束无法同时满足（隐藏格均须为正整数），"
            "不存在可行填表方案"
        )

    results: list[dict[str, Any]] = []
    for i, j in hidden:
        min_extra = _extreme_flow(
            hidden, row_demand, col_demand, residual_total, (i, j), +1
        )
        max_extra = _extreme_flow(
            hidden, row_demand, col_demand, residual_total, (i, j), -1
        )
        # 还原回“含下界 1”的真实隐藏格取值。
        min_value = 1 + min_extra
        max_value = 1 + max_extra
        results.append(
            {
                "row": i,
                "column": j,
                "min_value": min_value,
                "max_value": max_value,
                "inferable": min_value == max_value,
            }
        )
    return results


def _extreme_flow(
    hidden: list[tuple[int, int]],
    row_demand: list[int],
    col_demand: list[int],
    total: int,
    target: tuple[int, int],
    sign: int,
) -> int:
    """返回 target 隐藏边在可行流中的最小（sign=+1）/最大（sign=-1）流量。"""
    graph, cell_edge = _build_network(
        hidden, row_demand, col_demand, total, target, sign
    )
    flow = _send_flow(graph, total)
    if flow < total:
        # 可行性已由调用前的统一检查保证，此处仅作防御。
        raise unprocessable("不存在可行填表方案")
    edge_index = cell_edge[target]
    ti = target[0]
    # graph[ti][edge_index] = [列节点, 剩余容量, 费用, 反向边下标]。
    residual = graph[ti][edge_index][1]
    return total - residual


def _min_cost_flow(
    hidden: list[tuple[int, int]],
    row_demand: list[int],
    col_demand: list[int],
    total: int,
    target: tuple[int, int] | None,
    sign: int,
) -> tuple[int, dict[tuple[int, int], int]]:
    """构造网络并发送总流量 total，返回 (实际流量, 隐藏边正向边下标表)。"""
    graph, cell_edge = _build_network(
        hidden, row_demand, col_demand, total, target, sign
    )
    return _send_flow(graph, total), cell_edge


def _build_network(
    hidden: list[tuple[int, int]],
    row_demand: list[int],
    col_demand: list[int],
    total: int,
    target: tuple[int, int] | None,
    target_sign: int,
) -> tuple[list[list[list[int]]], dict[tuple[int, int], int]]:
    """构造最小费用流网络。

    源 -> 每行（容量 = 行需求，费用 0），行 -> 列（每隐藏边一条，
    容量 total，费用 0；target 边费用为 target_sign），列 -> 汇
    （容量 = 列需求，费用 0）。
    """
    rows = len(row_demand)
    cols = len(col_demand)
    source = rows + cols
    sink = source + 1
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(frm: int, to: int, capacity: int, cost: int) -> None:
        graph[frm].append([to, capacity, cost, len(graph[to])])
        graph[to].append([frm, 0, -cost, len(graph[frm]) - 1])

    for i, demand in enumerate(row_demand):
        if demand:
            add_edge(source, i, demand, 0)
    for j, demand in enumerate(col_demand):
        if demand:
            add_edge(rows + j, sink, demand, 0)
    cell_edge: dict[tuple[int, int], int] = {}
    for i, j in hidden:
        cost = target_sign if target == (i, j) else 0
        cell_edge[(i, j)] = len(graph[i])
        add_edge(i, rows + j, total, cost)
    return graph, cell_edge


def _send_flow(graph: list[list[list[int]]], total: int) -> int:
    """逐最短路（SPFA）增广，发送至多 total 流量，返回实际发送量。

    初始网络的费用只有 0 与同号的 target 费用，不存在负费用环；
    增广后残余网络中的负费用边只是已有正向边的反向边，标准连续最短路
    算法依然求得最小费用流。
    """
    source = len(graph) - 2
    sink = len(graph) - 1
    flow = 0
    while flow < total:
        dist = [float("inf")] * len(graph)
        prev_node = [-1] * len(graph)
        prev_edge = [-1] * len(graph)
        in_queue = [False] * len(graph)
        dist[source] = 0
        queue: deque[int] = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            for edge_index, edge in enumerate(graph[node]):
                if edge[1] > 0 and dist[node] + edge[2] < dist[edge[0]]:
                    dist[edge[0]] = dist[node] + edge[2]
                    prev_node[edge[0]] = node
                    prev_edge[edge[0]] = edge_index
                    if not in_queue[edge[0]]:
                        queue.append(edge[0])
                        in_queue[edge[0]] = True
        if dist[sink] == float("inf"):
            break

        pushed = total - flow
        node = sink
        while node != source:
            edge = graph[prev_node[node]][prev_edge[node]]
            pushed = min(pushed, edge[1])
            node = prev_node[node]
        flow += pushed
        node = sink
        while node != source:
            edge = graph[prev_node[node]][prev_edge[node]]
            edge[1] -= pushed
            graph[node][edge[3]][1] += pushed
            node = prev_node[node]
    return flow
