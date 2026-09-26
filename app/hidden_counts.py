"""已发布隐藏计数表的反推审计：精确求每个隐藏格的最小/最大可取正整数。

调用方提交一张 2～4 行、2～4 列的已发布计数表：公开格为正整数，隐藏格
为 ``null``；同时公开行合计与列合计（可直接使用
``/suppress-count-table`` 的输出）。把每个隐藏格看作未知正整数 x_ij，
公开格视为已知量，则：

    对每行 i：  sum_{隐藏列 j} x_ij = 行合计 - 行内公开格之和 = a_i
    对每列 j：  sum_{隐藏行 i} x_ij = 列合计 - 列内公开格之和 = b_j

这是带**边容量下界 1** 的行-列运输可行流问题（每行视为源端、每列视为
汇端）。表格至多 4×4（16 个未知量），使用 Dinic 最大流判定给定附加
约束下是否存在可行填表方案；判定复杂度只与 10 个节点、16 条格边的网络
结构有关，与计数取值大小无关（不枚举数值）。

对单个隐藏格 (r, c)：

- 精确最大值 = 最大的 v（1 <= v <= min(a_r, b_c)）使得
  “存在可行填法且 x_rc >= v”（在该格边设置下界 v）；
- 精确最小值 = 最小的 v 使得“存在可行填法且 x_rc <= v”
  （在该格边设置上界 v）。

两个判定关于 v 都单调，故用二分查找。隐藏格在所有可行填法中取值恒相同
（最小值 == 最大值）时标记为可反推格 ``inferable``。整个表不存在任何
可行填法时返回 422。
"""

from __future__ import annotations

from typing import Any

from .anonymizer import unprocessable

# 矩阵行数与列数的闭区间限制。
MIN_DIMENSION = 2
MAX_DIMENSION = 4


def prepare_table(raw_table: Any) -> list[list[int | None]]:
    """校验已发布计数表，返回公开格为正整数、隐藏格为 None 的矩形矩阵。

    矩阵必须 2～4 行、2～4 列且各行等长；公开格必须是正整数（拒绝布尔、
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
                    f"{label}的每格必须是正整数或 null（隐藏格）"
                )
            if value < 1:
                raise unprocessable(f"{label}的公开格必须是正整数（>= 1）")
            row.append(value)
        matrix.append(row)
    return matrix


def prepare_totals(
    raw_totals: Any, expected_length: int, name: str, dimension: str
) -> list[int]:
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


def _build_and_check(
    rows: int,
    cols: int,
    hidden: list[tuple[int, int]],
    caps: dict[tuple[int, int], int],
    row_remain: list[int],
    col_remain: list[int],
    target: tuple[int, int] | None,
    lower: int | None,
    upper: int | None,
) -> bool:
    """按带下界的行-列运输流建网，判定附加约束下是否存在可行填表方案。"""
    # 源=rows+cols，汇=源+1；超源/超汇用于下界边的循环转化。
    source = rows + cols
    sink = source + 1
    demands = [0] * (sink + 3)

    def bounded_edge(frm: int, to: int, low: int, high: int) -> None:
        # 有上下界的边：先预置 low 的流量，差额入残余网络。
        if low > high:
            raise _Infeasible
        demands[frm] -= low
        demands[to] += low
        if high > low:
            edges[frm].append([to, high - low, len(edges[to])])
            edges[to].append([frm, 0, len(edges[frm]) - 1])

    # 每个判定都重新建网：网络至多 11 个节点、十余条边。
    edges: list[list[list[int]]] = [[] for _ in range(rows + cols + 4)]

    # 行残量必须恰好由该行隐藏格凑出：源->行 为等式边（下界=上界）。
    for i in range(rows):
        bounded_edge(source, i, row_remain[i], row_remain[i])
    # 列->汇 同理为等式边。
    for j in range(cols):
        bounded_edge(rows + j, sink, col_remain[j], col_remain[j])

    for cell in hidden:
        i, j = cell
        edge_low = 1
        edge_high = caps[cell]
        if cell == target:
            if lower is not None:
                edge_low = lower
            if upper is not None:
                edge_high = upper
        bounded_edge(i, rows + j, edge_low, edge_high)

    # 汇->源 的回流边把“源到汇的固定流量”问题转化为循环流问题；
    # 流量只经过隐藏格，故下界为隐藏格残量总和（= 两侧残量总和）。
    hidden_total = sum(row_remain)
    bounded_edge(sink, source, hidden_total, hidden_total)

    super_source = sink + 1
    super_sink = sink + 2
    required = 0
    for node, demand in enumerate(demands):
        if demand > 0:
            edges[super_source].append([node, demand, len(edges[node])])
            edges[node].append(
                [super_source, 0, len(edges[super_source]) - 1]
            )
            required += demand
        elif demand < 0:
            edges[node].append(
                [super_sink, -demand, len(edges[super_sink])]
            )
            edges[super_sink].append(
                [node, 0, len(edges[node]) - 1]
            )

    return _max_flow(edges, super_source, super_sink) == required


def audit_hidden_counts(
    table: list[list[int | None]],
    row_totals: list[int],
    column_totals: list[int],
) -> list[dict[str, int | bool]]:
    """返回每个隐藏格（行优先）的精确取值区间与可反推标记。

    无任何可行填表方案时抛 422。结果每项含
    ``row/column/min_count/max_count/inferable``。
    """
    rows = len(table)
    cols = len(table[0])

    if sum(row_totals) != sum(column_totals):
        raise unprocessable(
            f"row_totals 的总和 {sum(row_totals)} 与 column_totals 的总和 "
            f"{sum(column_totals)} 必须相等"
        )

    hidden = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if table[i][j] is None
    ]

    # 各行列扣除公开格后的“隐藏格残量”：必须由该行/列的隐藏正整数凑出。
    row_remain = [
        row_totals[i] - sum(
            table[i][j] for j in range(cols) if table[i][j] is not None
        )
        for i in range(rows)
    ]
    col_remain = [
        column_totals[j] - sum(
            table[i][j] for i in range(rows) if table[i][j] is not None
        )
        for j in range(cols)
    ]
    if any(value < 0 for value in row_remain) or any(
        value < 0 for value in col_remain
    ):
        raise unprocessable(
            "不存在可行填表方案：某行/列的公开格之和超过了公开合计"
        )

    # 隐藏格 x_ij 的固有上界：它既不能超过本行残量也不能超过本列残量。
    caps = {
        (i, j): min(row_remain[i], col_remain[j]) for (i, j) in hidden
    }

    def feasible(
        target: tuple[int, int] | None,
        lower: int | None = None,
        upper: int | None = None,
    ) -> bool:
        """判定是否存在可行填法；target 格可附加精确下界/上界约束。"""
        try:
            return _build_and_check(
                rows,
                cols,
                hidden,
                caps,
                row_remain,
                col_remain,
                target,
                lower,
                upper,
            )
        except _Infeasible:
            # 强制下界超过固有上界等上下界矛盾：直接判不可行。
            return False

    if not feasible(None):
        raise unprocessable(
            "不存在可行填表方案：无法用各隐藏格的正整数同时满足公开格与"
            "全部行列合计"
        )

    results: list[dict[str, int | bool]] = []
    for cell in hidden:
        i, j = cell
        limit = caps[cell]

        # 最大值：最大的 v 使“x_rc >= v”可行（下界二分，可行域 1..max）。
        lo, hi = 1, limit
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(cell, lower=mid):
                lo = mid
            else:
                hi = mid - 1
        max_count = lo

        # 最小值：最小的 v 使“x_rc <= v”可行（上界二分，可行域 min..limit）。
        lo, hi = 1, limit
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(cell, upper=mid):
                hi = mid
            else:
                lo = mid + 1
        min_count = lo

        results.append(
            {
                "row": i,
                "column": j,
                "min_count": min_count,
                "max_count": max_count,
                "inferable": min_count == max_count,
            }
        )
    return results


class _Infeasible(Exception):
    """建网阶段即发现上下界矛盾（如强制下界超过固有上界）。"""


def _max_flow(graph: list[list[list[int]]], source: int, sink: int) -> int:
    """Dinic 求整数最大流。边结构：[终点, 剩余容量, 反向边下标]。"""

    def build_levels() -> list[int]:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for edge in graph[node]:
                to, capacity, _ = edge
                if capacity > 0 and levels[to] < 0:
                    levels[to] = levels[node] + 1
                    queue.append(to)
        return levels

    def send(node: int, pushed: int, levels: list[int], cursor: list[int]) -> int:
        if node == sink:
            return pushed
        while cursor[node] < len(graph[node]):
            edge = graph[node][cursor[node]]
            to, capacity, reverse = edge
            if capacity > 0 and levels[to] == levels[node] + 1:
                flow = send(to, min(pushed, capacity), levels, cursor)
                if flow:
                    edge[1] -= flow
                    graph[to][reverse][1] += flow
                    return flow
            cursor[node] += 1
        return 0

    total = 0
    while True:
        levels = build_levels()
        if levels[sink] < 0:
            return total
        cursor = [0] * len(graph)
        while True:
            pushed = send(source, 10**40, levels, cursor)
            if not pushed:
                break
            total += pushed
