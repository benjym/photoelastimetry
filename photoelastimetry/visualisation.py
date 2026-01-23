import sys

import numpy as np


def ascii_boundary_conditions(
    boundary_mask, boundary_values, title="Boundary Conditions", downsample_factor=None
):
    """
    Generate an ASCII representation of the boundary conditions.

    Parameters
    ----------
    boundary_mask : ndarray (bool)
        [H, W] boolean mask of active boundary pixels.
    boundary_values : dict
        Dictionary with keys 'xx', 'yy', 'xy', containing [H, W] arrays or scalars.
        NaN values indicate 'unconstrained' at that pixel for that component.
    title : str
        Title for the plot.
    downsample_factor : int, optional
        Factor to downsample grid for printing. If None, auto-calculated to fit ~80 chars width.

    Returns
    -------
    str
        ASCII string representation.
    """
    H, W = boundary_mask.shape

    # 1. Determine downsampling
    if downsample_factor is None:
        downsample_factor = max(1, W // 60, H // 40)

    h_small = (H + downsample_factor - 1) // downsample_factor
    w_small = (W + downsample_factor - 1) // downsample_factor

    # Create grid for simplified view
    # We will prioritize 'True' in the mask during downsampling (max pool)
    # Actually, standard striding is safer for visualization structure
    # mask_small = boundary_mask[::downsample_factor, ::downsample_factor]

    # Better: Downsample carefully. If any pixel in block is boundary, show it.
    grid_chars = np.full((h_small, w_small), " ", dtype=object)

    # Standardize values inputs
    components = ["xx", "yy", "xy"]
    val_maps = {}
    for c in components:
        if c in boundary_values:
            v = boundary_values[c]
            if np.isscalar(v):
                v = np.full((H, W), v)
            val_maps[c] = v
        else:
            val_maps[c] = np.full((H, W), np.nan)

    # Icons
    # Pinned/Clamped (All fixed): #
    # Roller Vert (X fixed, Y free?): |
    # Roller Horz (Y fixed, X free?): =
    # Shear only: S
    # Free surface (Zero traction): 0
    # Custom Load: L

    # Let's think in terms of local constraints
    # C = {xx, yy, xy}
    # For each block
    for r in range(h_small):
        for c in range(w_small):
            # Extract block
            r0, r1 = r * downsample_factor, min((r + 1) * downsample_factor, H)
            c0, c1 = c * downsample_factor, min((c + 1) * downsample_factor, W)

            block_mask = boundary_mask[r0:r1, c0:c1]
            if not np.any(block_mask):
                # Check if it's inside or outside?
                # Usually we just want to see boundaries.
                grid_chars[r, c] = "."  # internal/empty
                continue

            # Find the most constrained pixel in the block to represent it
            # Or just take center/first
            # Let's take the first masked pixel found
            y_local, x_local = np.where(block_mask)
            py, px = r0 + y_local[0], c0 + x_local[0]

            constraints = []
            values = []

            for comp in components:
                val = val_maps[comp][py, px]
                if not np.isnan(val):
                    constraints.append(comp)
                    values.append(val)

            # Symbol Logic
            is_xx = "xx" in constraints
            is_yy = "yy" in constraints
            is_xy = "xy" in constraints

            sym = "?"

            if is_xx and is_yy and is_xy:
                sym = "■"  # Clamped / Fully Known
            elif is_xx and is_yy:
                sym = "+"  # Bi-axial normal
            elif is_xx and is_xy:
                sym = "⦶"  # Vertical constraint?
            elif is_yy and is_xy:
                sym = "⦷"  # Horizontal constraint?
            elif is_xx:
                sym = "|"  # XX fixed (Normal X)
            elif is_yy:
                sym = "-"  # YY fixed (Normal Y)
            elif is_xy:
                sym = "x"  # Shear fixed
            else:
                sym = "o"  # Masked but no constraints? (Free?)

            # Special check for Zero (Free Surface) vs Non-Zero (Load)
            # If all constrained values are approx zero -> Free Surface
            # We differentiate 'Fixed to 0' from 'Fixed to Value'
            all_zero = True
            for v in values:
                if abs(v) > 1e-6:  # Tolerance
                    all_zero = False
                    break

            # If it's a "Free Surface" (Traction = 0), key components are usually Normal+Shear=0
            # But here we visualize components.
            # Let's denote Non-Zero loads with bold or different char?
            # Terminal bold: \033[1m ... \033[0m
            if not all_zero:
                # Load
                sym = f"\033[91m{sym}\033[0m"  # Red for Load
            else:
                # Zero constraint (Support/FreeSurface)
                sym = f"\033[92m{sym}\033[0m"  # Green for Zero

            grid_chars[r, c] = sym

    # Convert grid to string
    lines = []
    lines.append(f"=== {title} ===")
    lines.append(f"Grid: {W}x{H} -> {w_small}x{h_small} (DS: {downsample_factor})")

    # Improved Legend
    lines.append("Legend (Colors):")
    lines.append("  \033[92mGREEN\033[0m : Zero Value (Support / Free Surface)")
    lines.append("  \033[91mRED  \033[0m : Non-Zero Value (Load / Displacement)")

    lines.append("Legend (Symbols):")
    lines.append(f"  ■ : Clamped (xx, yy, xy fixed)")
    lines.append(f"  + : Bi-axial Normal (xx, yy fixed)")
    lines.append(f"  ⦶ : Vertical Roller / Side (xx, xy fixed)")
    lines.append(f"  ⦷ : Horizontal Roller / Top/Bot (yy, xy fixed)")
    lines.append(f"  | : Normal X fixed (xx)")
    lines.append(f"  - : Normal Y fixed (yy)")
    lines.append(f"  x : Shear fixed (xy)")
    lines.append(f"  o : Unconstrained (Masked but Free)")

    lines.append("-" * (w_small + 2))

    for r in range(h_small):
        row_str = "".join(grid_chars[r, :])
        lines.append(f" {row_str} ")

    lines.append("-" * (w_small + 2))
    return "\n".join(lines)


def print_boundary_conditions(boundary_mask, boundary_values):
    print(ascii_boundary_conditions(boundary_mask, boundary_values))


if __name__ == "__main__":
    # Test
    H, W = 20, 40
    mask = np.zeros((H, W), dtype=bool)
    mask[0, :] = True  # Top
    mask[-1, :] = True  # Bottom
    mask[:, 0] = True  # Left

    vals = {
        "yy": np.full((H, W), np.nan),
        "xx": np.full((H, W), np.nan),
        "xy": np.full((H, W), np.nan),
    }

    # Top: yy=0, xy=0
    vals["yy"][0, :] = 0
    vals["xy"][0, :] = 0

    # Bottom: Loaded yy=100
    vals["yy"][-1, :] = 100

    # Left: xx=0
    vals["xx"][:, 0] = 0

    print(ascii_boundary_conditions(mask, vals, downsample_factor=1))
