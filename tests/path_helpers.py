"""Grid expansion for readable fixture expectations."""
def expand_vertices(vertices):
    """Manual editing accepts corner coordinates, expanding straight segments."""
    cells = []
    for vertex in vertices:
        if len(vertex) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in vertex):
            raise ValueError("每个坐标必须是两个整数：[列, 行]")
        target = tuple(vertex)
        if not cells:
            cells.append(target)
            continue
        x, y = cells[-1]
        if x != target[0] and y != target[1] and abs(x-target[0]) != abs(y-target[1]):
            raise ValueError("相邻拐点必须横向、纵向或 45 度斜向相连")
        dx = (target[0] > x) - (target[0] < x)
        dy = (target[1] > y) - (target[1] < y)
        while (x, y) != target:
            x, y = x + dx, y + dy
            cells.append((x, y))
    if not cells:
        raise ValueError("路径不能为空")
    return cells

