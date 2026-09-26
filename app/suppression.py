"""观测计数表发布前的单元格隐藏：主抑制 + 数量最少的次抑制。

直接公开行列合计时，严格低于阈值 ``threshold`` 的格必须隐藏（主抑制，
亦称必须隐藏格）。若只隐藏这些格，攻击者可借助公开的行列合计把隐藏
值反推出来。把表格视为一张“行-列二部图”：每个格是连接其所在行节点
与列节点的一条边。由行列合计列线性方程可知，一条被隐藏的边只有位于
某个环上时取值才无法被唯一确定（等价判定：删去该边后其两端仍连通）；
孤立悬挂的隐藏边（桥）的值可被逐行逐列相减直接还原。

因此在“其余格”（计数值 >= 阈值、本不需隐藏的格）中选择数量最少的
补充隐藏格（次抑制），使每个必须隐藏格都位于全部隐藏格构成的环上；
当数量相同的方案不止一个时，按补充格行优先坐标序列的字典序取最小。

矩阵至多 4×4（16 格），候选补充格至多 16 个，直接按补充格数量递增、
同数量按行优先组合的字典序枚举，第一个可行方案即数量最少且字典序
最小的方案。矩阵至少 2×2 时，补充全部其余格后图为完全二部图，每条
边都在 4-环上，故对合法输入必存在可行解。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .anonymizer import unprocessable

# 矩阵行数与列数的闭区间限制。
MIN_DIMENSION = 2
MAX_DIMENSION = 4


def prepare_counts(raw_counts: Any) -> list[list[int]]:
    """校验计数矩阵，返回正整数矩阵（2～4 行、2～4 列的矩形）。

    每格必须是正整数（拒绝布尔、浮点与字符串）；所有行必须等长。
    """
    if not isinstance(raw_counts, list) or not raw_counts:
        raise unprocessable("counts 必须是至少包含 2 行的非空数组")
    if not MIN_DIMENSION <= len(raw_counts) <= MAX_DIMENSION:
        raise unprocessable(
            f"counts 必须包含 {MIN_DIMENSION}～{MAX_DIMENSION} 行，"
            f"当前为 {len(raw_counts)} 行"
        )

    matrix: list[list[int]] = []
    width: int | None = None
    for index, raw_row in enumerate(raw_counts):
        label = f"counts 第 {index + 1} 行"
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
        row: list[int] = []
        for value in raw_row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise unprocessable(f"{label}的每格计数必须是整数")
            if value < 1:
                raise unprocessable(f"{label}的每格计数必须是正整数（>= 1）")
            row.append(value)
        matrix.append(row)
    return matrix


def solve_suppression(
    counts: list[list[int]], threshold: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """返回 (必须隐藏格坐标, 补充隐藏格坐标)，均按行优先排序。

    必须隐藏格为计数值严格小于阈值的格；补充隐藏格从其余格中选出，
    数量最少且行优先坐标序列字典序最小，使每个必须隐藏格在“必须 +
    补充”全部隐藏格构成的行-列二部图中都位于某个环上。没有必须隐藏
    格时不需要任何补充隐藏。
    """
    rows = len(counts)
    cols = len(counts[0])

    # 必须隐藏（主抑制）：严格低于阈值；候选补充（次抑制）：其余格。
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
    if not required:
        return required, []

    required_edges = set(required)
    # 候选按行优先排列：combinations 按元组字典序产出，首个可行方案即
    # “数量最少、同数行优先序列字典序最小”的方案。
    for size in range(len(candidates) + 1):
        for chosen in combinations(candidates, size):
            edges = required_edges | set(chosen)
            adjacency = _build_adjacency(edges, rows, cols)
            if all(
                _edge_on_cycle(edge, adjacency, rows) for edge in required
            ):
                return required, list(chosen)

    # 理论不可达：矩阵 >= 2×2，取全部候选格后图为完全二部图，必有环。
    raise unprocessable("不存在使每个必须隐藏格都位于环上的补充隐藏方案")


def _build_adjacency(
    edges: set[tuple[int, int]], rows: int, cols: int
) -> list[list[int]]:
    """由格边集合构造二部图邻接表：0～rows-1 为行节点，其后为列节点。"""
    adjacency: list[list[int]] = [[] for _ in range(rows + cols)]
    for i, j in edges:
        u = i
        v = rows + j
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency


def _edge_on_cycle(
    edge: tuple[int, int], adjacency: list[list[int]], rows: int
) -> bool:
    """判定边是否属于某个环：删去该边（两个方向均不可走）后两端仍连通。"""
    i, j = edge
    start = i
    target = rows + j

    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            # 排除被判定的边本身（无向边的两个方向）。
            if (node == start and neighbor == target) or (
                node == target and neighbor == start
            ):
                continue
            if neighbor not in seen:
                if neighbor == target:
                    return True
                seen.add(neighbor)
                stack.append(neighbor)
    return False
