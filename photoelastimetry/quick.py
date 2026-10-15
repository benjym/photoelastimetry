"""First-branch optical inversion without fringe resolution or stress fitting."""

import argparse
from pathlib import Path

import numpy as np

from photoelastimetry.io import collapse_duplicate_green_channel, load_image, save_image, split_channels
from photoelastimetry.seeding import invert_wrapped_retardance


def _stack(data, raw_mosaic=False):
    data = np.asarray(data, dtype=float)
    if raw_mosaic:
        if data.ndim != 2 or any(n % 4 for n in data.shape):
            raise ValueError("Raw mosaics must be 2D with dimensions divisible by 4.")
        data = collapse_duplicate_green_channel(split_channels(data))
    elif data.ndim == 3 and data.shape[-1] == 4:
        data = data[:, :, None, :]
    if data.ndim != 4 or data.shape[-1] != 4 or 0 in data.shape:
        raise ValueError("Expected [H, W, C, 4] or monochrome [H, W, 4], ordered 0, 45, 90, 135 degrees.")
    return data


def first_period_inversion(data, input_stokes=(0.0, 0.0, 1.0), bright_field=None, dark_field=None):
    """Return per-colour retardance and direction, both [H, W, C] in radians.

    Accept ideal fully polarised linear or circular illumination. Retardance uses
    the lowest branch: [0, pi] for linear, [0, pi/2] for circular. Linear direction
    is ambiguous modulo pi/2; circular direction is modulo pi. Undefined pixels
    (including linear isoclinics) are NaN. References must match the input shape;
    bright fields should measure uniform, unpolarised detector illumination.
    """
    data = _stack(data)
    source = np.asarray(input_stokes, dtype=float)
    if source.shape != (3,) or not np.all(np.isfinite(source)) or not np.isclose(np.linalg.norm(source), 1):
        raise ValueError("input_stokes must be a fully polarised 3-component unit vector.")
    circular = np.isclose(abs(source[2]), 1) and np.allclose(source[:2], 0)
    if not circular and not np.isclose(source[2], 0):
        raise ValueError("Only pure linear or circular illumination is supported.")

    dark = 0.0
    if dark_field is not None:
        dark = _stack(dark_field)
        if dark.shape != data.shape:
            raise ValueError("Dark field must match the input shape.")
    corrected = data - dark
    if bright_field is not None:
        bright = _stack(bright_field)
        if bright.shape != data.shape:
            raise ValueError("Bright field must match the input shape.")
        response = bright - dark
        corrected = np.divide(
            corrected, response, out=np.full_like(data, np.nan), where=np.isfinite(response) & (response > 0)
        )

    valid = np.all(np.isfinite(corrected) & (corrected >= 0), axis=-1)
    intensity = corrected.sum(axis=-1) / 2
    valid &= intensity > 0
    stokes = np.stack((corrected[..., 0] - corrected[..., 2], corrected[..., 1] - corrected[..., 3]), axis=-1)
    stokes = np.divide(stokes, intensity[..., None], out=np.zeros_like(stokes), where=valid[..., None])
    # Invert each colour independently by adding a singleton wavelength dimension.
    theta, delta = invert_wrapped_retardance(stokes[..., None, :], source)
    delta = delta[..., 0]
    if circular:
        signal = np.linalg.norm(stokes, axis=-1)
        theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    else:
        signal = np.linalg.norm(stokes - source[:2], axis=-1)
        alpha = 0.5 * np.arctan2(source[1], source[0])
        valid &= np.abs(np.sin(2 * (theta - alpha))) > 1e-3
        theta = theta % (np.pi / 2)
    direction_valid = valid & (signal > 1e-10)
    # On the lowest circular branch, zero signal means zero retardation, but
    # cannot constrain direction. Linear zero signal can also be an isoclinic.
    if not circular:
        valid &= signal > 1e-10
    return np.where(valid, delta, np.nan), np.where(direction_valid, theta, np.nan)


def image_to_retardation(
    input_filename,
    output_prefix=None,
    output_format="tiff",
    input_stokes=(0, 0, 1),
    bright_field=None,
    dark_field=None,
    raw_mosaic=False,
):
    """Load a polarimetric image and write retardation/direction TIFFs or PNG previews."""
    output_format = output_format.lower()
    if output_format not in {"tiff", "tif", "png"}:
        raise ValueError("Output format must be tiff, tif or png.")

    def read(path):
        data, _ = load_image(str(path))
        return _stack(data, raw_mosaic=raw_mosaic or str(path).lower().endswith(".raw"))

    delta, theta = first_period_inversion(
        read(input_filename),
        input_stokes,
        bright_field=read(bright_field) if bright_field is not None else None,
        dark_field=read(dark_field) if dark_field is not None else None,
    )
    prefix = Path(output_prefix) if output_prefix is not None else Path(input_filename).with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if output_format in {"tiff", "tif"}:
        for name, values in (("retardation", delta), ("direction", theta)):
            save_image(f"{prefix}_{name}.{output_format}", values)
    else:
        from matplotlib import colormaps
        from matplotlib.image import imsave

        circular = np.isclose(abs(input_stokes[2]), 1)
        delta_max = np.pi / 2 if circular else np.pi
        for channel in range(delta.shape[-1]):
            for name, values, low, high in (
                ("retardation", delta, 0, delta_max),
                ("direction", theta, -np.pi / 2 if circular else 0, np.pi / 2),
            ):
                cmap = colormaps["gray"].copy()
                cmap.set_bad((0, 0, 0, 0))
                imsave(
                    f"{prefix}_{name}_c{channel}.png",
                    np.ma.masked_invalid(values[..., channel]),
                    cmap=cmap,
                    vmin=low,
                    vmax=high,
                )
    return delta, theta


def cli_image_to_retardation():
    parser = argparse.ArgumentParser(
        description="Quick first-branch retardation/direction inversion (no unwrapping)."
    )
    parser.add_argument("input_filename", help="Polarisation TIFF/NPY stack or raw camera frame")
    parser.add_argument("-o", "--output-prefix", help="Output path prefix (default: input without extension)")
    parser.add_argument(
        "-f", "--format", choices=("tiff", "tif", "png"), default="tiff", dest="output_format"
    )
    parser.add_argument(
        "--input-stokes",
        nargs=3,
        type=float,
        default=(0, 0, 1),
        metavar=("S1", "S2", "S3"),
        help="Ideal illumination state; default: 0 0 1 (circular). Linear: 1 0 0",
    )
    parser.add_argument(
        "--bright-field", help="Uniform unpolarised flat-field image, matching the input shape"
    )
    parser.add_argument("--dark-field", help="Dark reference image, matching the input shape")
    parser.add_argument(
        "--raw-mosaic", action="store_true", help="Treat 2D input/reference images as 4x4 sensor mosaics"
    )
    args = vars(parser.parse_args())
    try:
        image_to_retardation(**args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli_image_to_retardation()
