"""Tests for PARALLEL grid allocation — disjoint trial partitions.

T1.1: allocate(strategy="PARALLEL", total_cells=24, worker_id=0, total_workers=4)
returns 6 cells with no overlap across workers.

Uses modular arithmetic: cell_index % total_workers == worker_id.
"""

from __future__ import annotations


class TestGridPartitioning:
    """Test that grid partitioning produces disjoint, exhaustive cell sets."""

    def test_disjoint_partitions_4_workers(self) -> None:
        """4 workers over 24 cells: each gets 6, no overlap."""
        from minivess.optimization.grid_partitioning import partition_grid_cells

        total_cells = 24
        total_workers = 4
        all_cells: list[list[int]] = []
        for worker_id in range(total_workers):
            cells = partition_grid_cells(
                total_cells=total_cells,
                worker_id=worker_id,
                total_workers=total_workers,
            )
            all_cells.append(cells)

        # Each worker gets exactly 6 cells
        for worker_id, cells in enumerate(all_cells):
            assert len(cells) == 6, (
                f"Worker {worker_id} got {len(cells)} cells, expected 6"
            )

        # Union covers all 24 cells
        union = set()
        for cells in all_cells:
            union.update(cells)
        assert union == set(range(24)), (
            f"Union of all partitions must be {{0..23}}, got {union}"
        )

        # Pairwise disjoint
        for i in range(total_workers):
            for j in range(i + 1, total_workers):
                overlap = set(all_cells[i]) & set(all_cells[j])
                assert not overlap, f"Workers {i} and {j} overlap on cells {overlap}"

    def test_disjoint_partitions_uneven_split(self) -> None:
        """7 cells across 3 workers: [0,3,6], [1,4], [2,5]."""
        from minivess.optimization.grid_partitioning import partition_grid_cells

        total_cells = 7
        total_workers = 3
        all_cells: list[list[int]] = []
        for worker_id in range(total_workers):
            cells = partition_grid_cells(
                total_cells=total_cells,
                worker_id=worker_id,
                total_workers=total_workers,
            )
            all_cells.append(cells)

        # Union covers all 7 cells
        union = set()
        for cells in all_cells:
            union.update(cells)
        assert union == set(range(7))

        # Worker 0 gets ceil(7/3)=3 cells, others get 2
        assert len(all_cells[0]) == 3
        assert len(all_cells[1]) == 2
        assert len(all_cells[2]) == 2

    def test_single_worker_gets_all(self) -> None:
        """1 worker gets all cells."""
        from minivess.optimization.grid_partitioning import partition_grid_cells

        cells = partition_grid_cells(total_cells=24, worker_id=0, total_workers=1)
        assert cells == list(range(24))

    def test_worker_id_out_of_range_raises(self) -> None:
        """worker_id >= total_workers must raise ValueError."""
        import pytest

        from minivess.optimization.grid_partitioning import partition_grid_cells

        with pytest.raises(ValueError, match="worker_id"):
            partition_grid_cells(total_cells=24, worker_id=4, total_workers=4)

    def test_zero_cells_returns_empty(self) -> None:
        """0 total cells: every worker gets empty list."""
        from minivess.optimization.grid_partitioning import partition_grid_cells

        cells = partition_grid_cells(total_cells=0, worker_id=0, total_workers=4)
        assert cells == []


class TestGridCellExpansion:
    """Test expanding cell indices to hyperparameter dicts."""

    def test_expand_cell_to_params(self) -> None:
        """Cell index 0 with 2 losses and 2 models → first combo."""
        from minivess.optimization.grid_partitioning import expand_grid_cell

        factors = {
            "model_family": ["dynunet", "segresnet"],
            "loss_name": ["dice_ce", "cbdice_cldice"],
        }
        params = expand_grid_cell(cell_index=0, factors=factors)
        assert isinstance(params, dict)
        assert "model_family" in params
        assert "loss_name" in params

    def test_expand_24_cells_unique(self) -> None:
        """24 cells from 4×3×2 factorial: all distinct."""
        from minivess.optimization.grid_partitioning import expand_grid_cell

        factors = {
            "model_family": ["dynunet", "segresnet", "sam3_vanilla", "vesselfm"],
            "loss_name": ["cbdice_cldice", "dice_ce", "dice_ce_cldice"],
            "aux_calibration": [True, False],
        }
        total = 4 * 3 * 2
        assert total == 24

        param_sets = []
        for i in range(total):
            p = expand_grid_cell(cell_index=i, factors=factors)
            param_sets.append(tuple(sorted(p.items())))

        # All 24 combos are unique
        assert len(set(param_sets)) == 24, (
            f"Expected 24 unique combos, got {len(set(param_sets))}"
        )

    def test_cell_index_out_of_range_raises(self) -> None:
        """Cell index >= product of factor sizes must raise ValueError."""
        import pytest

        from minivess.optimization.grid_partitioning import expand_grid_cell

        factors = {"a": [1, 2], "b": [3, 4]}
        with pytest.raises(ValueError, match="cell_index"):
            expand_grid_cell(cell_index=4, factors=factors)
