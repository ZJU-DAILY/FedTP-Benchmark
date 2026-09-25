import math


GRID_HW_MAP = {
    "TaxiBJ": (32, 32),
    "TaxiNYC": (15, 5),
    "BikeNYC": (16, 8),
}


def grid_hw_for_dataset(dataset_name):
    for key, hw in GRID_HW_MAP.items():
        if key in str(dataset_name):
            return hw
    raise ValueError(f"Unknown grid dataset shape for {dataset_name}")


def _contiguous_slices(length, num_parts):
    quotient, remainder = divmod(length, num_parts)
    slices = []
    start = 0
    for idx in range(num_parts):
        size = quotient + (1 if idx < remainder else 0)
        end = start + size
        slices.append((start, end))
        start = end
    return slices


def _factor_pairs(num_parts):
    pairs = []
    for rows in range(1, int(math.sqrt(num_parts)) + 1):
        if num_parts % rows == 0:
            cols = num_parts // rows
            pairs.append((rows, cols))
            if rows != cols:
                pairs.append((cols, rows))
    return pairs


def _best_rectangular_factor(h, w, num_parts):
    best_pair = None
    best_score = None
    for rows, cols in _factor_pairs(num_parts):
        if rows > h or cols > w:
            continue
        cell_h = h / rows
        cell_w = w / cols
        aspect_score = abs(math.log(max(cell_h, 1e-8) / max(cell_w, 1e-8)))
        balance_score = abs(rows - cols) * 1e-3
        score = aspect_score + balance_score
        if best_score is None or score < best_score:
            best_pair = (rows, cols)
            best_score = score

    if best_pair is None:
        raise ValueError(
            f"Cannot split grid {h}x{w} into {num_parts} non-empty rectangular clients."
        )
    return best_pair


def grid_rectangular_split(dataset_name, num_clients):
    if int(num_clients) <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")

    h, w = grid_hw_for_dataset(dataset_name)
    if int(num_clients) == 1:
        return [list(range(h * w))]

    row_parts, col_parts = _best_rectangular_factor(h, w, int(num_clients))
    row_slices = _contiguous_slices(h, row_parts)
    col_slices = _contiguous_slices(w, col_parts)

    nodes_per = []
    for row_start, row_end in row_slices:
        for col_start, col_end in col_slices:
            block = []
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    block.append(r * w + c)
            nodes_per.append(block)

    if len(nodes_per) != int(num_clients):
        raise RuntimeError(
            f"Expected {num_clients} client partitions, got {len(nodes_per)}"
        )
    if any(len(block) == 0 for block in nodes_per):
        raise RuntimeError("Grid rectangular split generated an empty client partition.")
    return nodes_per
