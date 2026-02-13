import argparse
import os

import json5
import numpy as np
from scipy.ndimage import gaussian_filter

import photoelastimetry.calibrate
import photoelastimetry.io
import photoelastimetry.optimise
import photoelastimetry.plotting
import photoelastimetry.seeding
from photoelastimetry.image import compute_retardance, simulate_four_step_polarimetry


def _normalise_wavelengths(wavelengths):
    """Return wavelengths as a 1D float array in meters."""
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim == 0:
        wl = wl.reshape(1)
    if np.max(wl) > 1e-6:
        wl = wl * 1e-9
    return wl


def _load_array(path):
    """Load an array from an image path via io.load_image."""
    data, _ = photoelastimetry.io.load_image(path)
    return data


def _merge_params_with_calibration(params, fallback_params=None):
    """
    Merge params with optional fallback and calibration profile values.

    Precedence order is:
    1) explicit params
    2) fallback params
    3) calibration profile for C/S_i_hat/wavelengths
    """
    merged = {}
    if fallback_params:
        merged.update(fallback_params)
    merged.update(params)

    calibration_file = merged.get("calibration_file")
    if calibration_file is None:
        return merged

    profile = photoelastimetry.calibrate.load_calibration_profile(calibration_file)
    for key in ("C", "S_i_hat", "wavelengths"):
        if key not in merged and key in profile:
            merged[key] = profile[key]
    merged["_calibration_profile"] = profile
    return merged


def _get_stress_components(stress_map, stress_order):
    """
    Extract (sigma_xx, sigma_yy, sigma_xy) from a stress map using the specified order.
    """
    if stress_map.ndim != 3 or stress_map.shape[2] != 3:
        raise ValueError(f"Stress map must have shape [H, W, 3], got {stress_map.shape}")

    if stress_order == "xx_yy_xy":
        sigma_xx = stress_map[:, :, 0]
        sigma_yy = stress_map[:, :, 1]
        sigma_xy = stress_map[:, :, 2]
    elif stress_order == "xy_yy_xx":
        sigma_xy = stress_map[:, :, 0]
        sigma_yy = stress_map[:, :, 1]
        sigma_xx = stress_map[:, :, 2]
    else:
        raise ValueError(f"Unsupported stress_order '{stress_order}'. Use 'xx_yy_xy' or 'xy_yy_xx'.")

    return sigma_xx, sigma_yy, sigma_xy


def image_to_stress(params, output_filename=None):
    """
    Convert photoelastic images to stress maps.

    This function processes raw photoelastic data to recover stress distribution maps
    using the stress-optic law and polarisation analysis.

    Args:
        params (dict): Configuration dictionary containing:
            - input_filename (str, optional): Path to input image file. If None, raw images are loaded from folderName.
            - folderName (str): Path to folder containing raw photoelastic images
            - crop (list, optional): Crop region as [x1, x2, y1, y2]
            - debug (bool): If True, display all channels for debugging
            - C (float): Stress-optic coefficient in 1/Pa
            - thickness (float): Sample thickness in meters
            - wavelengths (list): List of wavelengths in nanometers
            - S_i_hat (list): Incoming normalised Stokes vector [S1_hat, S2_hat, S3_hat]
            - seeding (dict, optional): Seeding controls (`enabled`, `n_max`, `sigma_max`)
            - top-level optimise options (optional): `knot_spacing`, `spline_degree`,
              `boundary_mask_file`, `boundary_values_files`, `boundary_weight`,
              `regularisation_weight` (or `regularization_weight`), `regularisation_order`,
              `external_potential_file`, `external_potential_gradient`, `max_iterations`,
              `tolerance`, `verbose`, `debug`
        output_filename (str, optional): Path to save the output stress map image.
            If None, the stress map is not saved. Defaults to None. Can also be specified in params.

    Returns:
        numpy.ndarray: 3D stress tensor map [H, W, 3] in Pascals.

    Notes:
        - Assumes incoming light is fully S1 polarised
        - Uses uniform stress-optic coefficient across all wavelengths
        - Assumes solid sample (NU = 1.0)
        - Wavelengths are automatically converted from nm to meters
    """

    params = _merge_params_with_calibration(params)

    if "folderName" in params:
        data, metadata = photoelastimetry.io.load_raw(params["folderName"])
    elif "input_filename" in params:
        data, metadata = photoelastimetry.io.load_image(params["input_filename"])
    else:
        raise ValueError("Either 'folderName' or 'input_filename' must be specified in params.")

    profile = params.get("_calibration_profile")
    if profile is not None:
        data = photoelastimetry.calibrate.apply_blank_correction(data, profile["blank_correction"])

    if params.get("crop") is not None:
        data = data[
            params["crop"][2] : params["crop"][3],
            params["crop"][0] : params["crop"][1],
            :,
            :,
        ]
        if params["debug"]:
            photoelastimetry.io.save_image("debug_cropped_image.tiff", data, metadata)

    if params["debug"]:
        # import matplotlib.pyplot as plt
        import tifffile

        tifffile.imwrite("debug_before_binning.tiff", data[:, :, 0, 0])

    if params.get("binning") is not None:
        binning = params["binning"]
        data = photoelastimetry.io.bin_image(data, binning)
        metadata["height"] //= binning
        metadata["width"] //= binning

    if params["debug"]:
        photoelastimetry.plotting.show_all_channels(data, metadata)

    if "C" not in params:
        raise ValueError("Missing stress-optic coefficient 'C'. Provide it directly or via calibration_file.")
    if "thickness" not in params:
        raise ValueError("Missing sample thickness 'thickness'.")
    if "wavelengths" not in params:
        raise ValueError("Missing wavelengths. Provide them directly or via calibration_file.")
    if "S_i_hat" not in params:
        raise ValueError("Missing S_i_hat. Provide it directly or via calibration_file.")

    C = params["C"]  # Stress-optic coefficients in 1/Pa
    L = params["thickness"]  # Thickness in m
    WAVELENGTHS = _normalise_wavelengths(params["wavelengths"])
    NU = 1.0  # Solid sample
    if isinstance(C, list) or isinstance(C, np.ndarray):
        C_VALUES = np.asarray(C, dtype=float)
    else:
        C_VALUES = np.array(
            [
                C,
                C,
                C,
            ],
            dtype=float,
        )  # Stress-optic coefficients in 1/Pa

    # Get incoming polarisation state from config
    S_I_HAT = np.array(params["S_i_hat"])
    # Ensure it's 3 elements for consistency
    if len(S_I_HAT) == 2:
        S_I_HAT = np.append(S_I_HAT, 0.0)  # Add S3_hat = 0 for backward compatibility

    if "solver" in params:
        raise ValueError(
            "`solver` is no longer supported. "
            "image_to_stress now always runs the optimise solver. "
            "Remove `solver` from params."
        )
    if "global_mean_stress" in params or "global_solver" in params:
        raise ValueError(
            "Nested solver config blocks (`global_mean_stress`, `global_solver`) are no longer supported. "
            "Move solver options to top-level params."
        )

    # Phase Decomposed Seeding
    seeding_config = params.get("seeding", {})
    n_max = seeding_config.get("n_max", 6)
    sigma_max = seeding_config.get("sigma_max", 10e6)

    print("Running phase decomposed seeding...")
    initial_stress_map = photoelastimetry.seeding.phase_decomposed_seeding(
        data,
        WAVELENGTHS,
        C_VALUES,
        NU,
        L,
        S_i_hat=S_I_HAT,
        sigma_max=sigma_max,
        n_max=n_max,
    )

    H, W = data.shape[:2]

    boundary_mask = None
    mask_file = params.get("boundary_mask_file")
    if mask_file is not None:
        if not os.path.exists(mask_file):
            raise ValueError(f"Boundary mask file not found: {mask_file}")
        boundary_mask = _load_array(mask_file) > 0
        if boundary_mask.shape != (H, W):
            raise ValueError(f"Boundary mask shape must be {(H, W)}, got {boundary_mask.shape}")

    boundary_values = None
    boundary_value_files = params.get("boundary_values_files")
    if boundary_value_files is not None:
        boundary_values = {}
        for key in ("xx", "yy", "xy"):
            if key not in boundary_value_files:
                continue
            value_file = boundary_value_files[key]
            if value_file is None:
                continue
            if not os.path.exists(value_file):
                raise ValueError(f"Boundary values file for '{key}' not found: {value_file}")
            boundary_values[key] = _load_array(value_file).astype(float)
            if boundary_values[key].shape != (H, W):
                raise ValueError(
                    f"Boundary values '{key}' shape must be {(H, W)}, got {boundary_values[key].shape}"
                )
        if len(boundary_values) == 0:
            boundary_values = None

    external_potential = None
    potential_file = params.get("external_potential_file")
    potential_gradient = params.get("external_potential_gradient")
    if potential_file is not None and potential_gradient is not None:
        raise ValueError("Use either external_potential_file or external_potential_gradient, not both.")
    if potential_file is not None:
        if not os.path.exists(potential_file):
            raise ValueError(f"External potential file not found: {potential_file}")
        external_potential = _load_array(potential_file).astype(float)
        if external_potential.shape != (H, W):
            raise ValueError(f"External potential shape must be {(H, W)}, got {external_potential.shape}")
    elif potential_gradient is not None:
        grad = np.asarray(potential_gradient, dtype=float)
        if grad.shape != (2,):
            raise ValueError(f"external_potential_gradient must be [dVdx, dVdy], got {potential_gradient}")
        y_idx, x_idx = np.indices((H, W), dtype=float)
        external_potential = grad[0] * x_idx + grad[1] * y_idx

    optimise_params = {}
    for key in (
        "knot_spacing",
        "spline_degree",
        "boundary_weight",
        "regularisation_weight",
        "regularisation_order",
        "max_iterations",
        "tolerance",
        "verbose",
        "debug",
    ):
        if key in params:
            optimise_params[key] = params[key]

    if "regularization_weight" in params and "regularisation_weight" not in optimise_params:
        optimise_params["regularisation_weight"] = params["regularization_weight"]

    initial_diff, initial_theta = photoelastimetry.optimise.stress_to_principal_invariants(initial_stress_map)

    bspline_wrapper, coeffs = photoelastimetry.optimise.recover_mean_stress(
        initial_diff,
        initial_theta,
        boundary_mask=boundary_mask,
        boundary_values=boundary_values,
        external_potential=external_potential,
        initial_stress_map=initial_stress_map,
        **optimise_params,
    )

    s_xx, s_yy, s_xy = bspline_wrapper.get_stress_fields(coeffs)
    if external_potential is not None:
        s_xx = s_xx + external_potential
        s_yy = s_yy + external_potential
    stress_map = np.stack([s_xx, s_yy, s_xy], axis=-1)

    if params.get("output_filename") is not None:
        output_filename = params["output_filename"]

    if output_filename is not None:
        photoelastimetry.io.save_image(output_filename, stress_map, metadata)

    return stress_map


def stress_to_image(params):
    """
    Convert stress field data to photoelastic fringe pattern image.

    This function loads stress field data, optionally applies Gaussian scattering,
    computes principal stresses and their orientations, calculates photoelastic
    retardation and fringe patterns, and saves the resulting visualization.

    Args:
        params (dict): Dictionary containing the following keys:
            - p_filename (str): Path to the photoelastimetry parameter file
            - stress_filename (str): Path to the stress field data file
            - scattering (float, optional): Gaussian filter sigma for scattering simulation.
              If falsy, no scattering is applied.
            - t (float): Thickness of the photoelastic material
            - lambda_light (float): Wavelength of light used in the experiment
            - C (float): Stress-optic coefficient of the material
            - output_filename (str, optional): Path for the output image.
              Defaults to "output.png" if not provided.

    Returns:
        None: The function saves the fringe pattern visualization to a file.

    Notes:
        - The stress field is expected to have components in the order [sigma_xy, sigma_yy, sigma_xx]
        - Principal stresses are computed using Mohr's circle equations
        - Isochromatic fringe intensity is calculated using sin²(δ/2)
        - Isoclinic angle represents the orientation of principal stresses
    """

    fallback_params = {}
    if "p_filename" in params:
        if not os.path.exists(params["p_filename"]):
            raise ValueError(f"Parameter file not found: {params['p_filename']}")
        with open(params["p_filename"], "r") as f:
            fallback_params = json5.load(f)

    merged_params = _merge_params_with_calibration(params, fallback_params=fallback_params)

    def get_param(name, default=None, aliases=()):
        names = (name,) + tuple(aliases)
        for key in names:
            if key in merged_params:
                return merged_params[key]
        return default

    stress_filename = get_param("stress_filename", aliases=("s_filename",))
    if stress_filename is None:
        raise ValueError("Missing stress map path. Provide 'stress_filename' (or legacy 's_filename').")

    stress_map, _ = photoelastimetry.io.load_image(stress_filename)
    stress_order = get_param("stress_order", default="xx_yy_xy")
    sigma_xx, sigma_yy, sigma_xy = _get_stress_components(stress_map, stress_order)

    scattering = float(get_param("scattering", default=0.0))
    if scattering > 0:
        sigma_xx = gaussian_filter(sigma_xx, sigma=scattering)
        sigma_xy = gaussian_filter(sigma_xy, sigma=scattering)
        sigma_yy = gaussian_filter(sigma_yy, sigma=scattering)

    wavelengths_cfg = get_param("wavelengths", aliases=("lambda_light",))
    if wavelengths_cfg is None:
        raise ValueError("Missing wavelengths. Provide 'wavelengths' or 'lambda_light'.")
    wavelengths = _normalise_wavelengths(wavelengths_cfg)

    C_cfg = get_param("C")
    if C_cfg is None:
        raise ValueError("Missing stress-optic coefficient 'C'.")
    C_values = np.asarray(C_cfg, dtype=float)
    if C_values.ndim == 0:
        C_values = np.full(wavelengths.shape, float(C_values))
    elif C_values.size == 1 and wavelengths.size > 1:
        C_values = np.full(wavelengths.shape, float(C_values.item()))
    elif C_values.size != wavelengths.size:
        raise ValueError(f"C must be scalar or length {wavelengths.size}, got length {C_values.size}.")

    thickness = get_param("thickness", aliases=("t",))
    if thickness is None:
        raise ValueError("Missing sample thickness. Provide 'thickness' or legacy key 't'.")
    thickness = float(thickness)

    nu = float(get_param("nu", default=1.0))
    S_i_hat = np.asarray(get_param("S_i_hat", default=[1.0, 0.0, 0.0]), dtype=float)
    if len(S_i_hat) == 2:
        S_i_hat = np.append(S_i_hat, 0.0)
    elif len(S_i_hat) != 3:
        raise ValueError(f"S_i_hat must have length 2 or 3, got {len(S_i_hat)}.")

    H, W = sigma_xx.shape
    synthetic_images = np.zeros((H, W, len(wavelengths), 4), dtype=np.float32)
    for i, (wl, C_val) in enumerate(zip(wavelengths, C_values)):
        I0, I45, I90, I135 = simulate_four_step_polarimetry(
            sigma_xx, sigma_yy, sigma_xy, C_val, nu, thickness, wl, S_i_hat
        )
        synthetic_images[:, :, i, 0] = I0
        synthetic_images[:, :, i, 1] = I45
        synthetic_images[:, :, i, 2] = I90
        synthetic_images[:, :, i, 3] = I135

    output_filename = get_param("output_filename", default="output.png")
    ext = os.path.splitext(output_filename)[1].lower()

    if ext in {".tiff", ".tif", ".npy", ".raw"}:
        photoelastimetry.io.save_image(output_filename, synthetic_images)
    elif ext in {".png", ".jpg", ".jpeg"}:
        delta = compute_retardance(sigma_xx, sigma_yy, sigma_xy, C_values[0], nu, thickness, wavelengths[0])
        fringe_intensity = np.sin(delta / 2) ** 2
        phi = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)
        photoelastimetry.plotting.plot_fringe_pattern(fringe_intensity, phi, filename=output_filename)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}' for stress_to_image. "
            "Use .tiff/.tif/.npy/.raw for stacks or .png/.jpg/.jpeg for a plot."
        )

    return synthetic_images


def cli_image_to_stress():
    """Command line interface for image_to_stress function."""
    parser = argparse.ArgumentParser(description="Convert photoelastic images to stress maps.")
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the JSON5 parameter file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output stress map image (optional).",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))
    image_to_stress(params, output_filename=args.output)


def cli_stress_to_image():
    """Command line interface for stress_to_image function."""
    parser = argparse.ArgumentParser(
        description="Convert stress field data to photoelastic fringe pattern image."
    )
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the JSON5 parameter file.",
    )
    args = parser.parse_args()

    params = json5.load(open(args.json_filename, "r"))
    stress_to_image(params)


def cli_calibrate():
    """Command line interface for calibration workflow."""
    parser = argparse.ArgumentParser(
        description="Calibrate photoelastimetry parameters from known-load data."
    )
    parser.add_argument(
        "json_filename",
        type=str,
        help="Path to the calibration JSON5 parameter file.",
    )
    args = parser.parse_args()

    with open(args.json_filename, "r") as f:
        params = json5.load(f)

    result = photoelastimetry.calibrate.run_calibration(params)
    print(f"Wrote calibration profile: {result['profile_file']}")
    print(f"Wrote calibration report: {result['report_file']}")
    print(f"Wrote calibration diagnostics: {result['diagnostics_file']}")


def demosaic_raw_image(input_file, metadata, output_prefix=None, output_format="tiff"):
    """
    De-mosaic a raw polarimetric image and save to TIFF stack or individual PNGs.

    This function takes a raw image from a polarimetric camera with a 4x4 superpixel
    pattern and splits it into separate channels for each color and polarisation angle.

    Args:
        input_file (str): Path to the raw image file.
        metadata (dict): Dictionary containing image metadata with keys:
            - width (int): Image width in pixels
            - height (int): Image height in pixels
            - dtype (str, optional): Data type ('uint8' or 'uint16')
        output_prefix (str, optional): Prefix for output files. If None, uses input
            filename without extension. Defaults to None.
        output_format (str, optional): Output format, either 'tiff' for a single
            TIFF stack or 'png' for individual PNG files. Defaults to 'tiff'.

    Returns:
        numpy.ndarray: De-mosaiced image stack of shape [H, W, 4, 4] where:
            - H, W are the de-mosaiced dimensions (1/4 of original)
            - First dimension 4: color channels (R, G1, G2, B)
            - Second dimension 4: polarisation angles (0°, 45°, 90°, 135°)

    Notes:
        - The raw image uses a 4x4 superpixel pattern with interleaved polarisation
          and color filters
        - Output TIFF stack has shape [H, W, 4, 4] with all channels
        - Output PNGs create 4 files (one per polarisation angle) with shape [H, W, 4]
          showing all color channels
    """
    # Read raw image
    data = photoelastimetry.io.read_raw(input_file, metadata)

    # De-mosaic into channels
    demosaiced = photoelastimetry.io.split_channels(data)

    # Keep only R, G1, B channels by removing G2
    demosaiced = demosaiced[:, :, [0, 1, 3], :]  # Keep R, G1, B

    # Determine output filename prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(input_file)[0]

    # Save based on format
    if output_format.lower() == "tiff":
        import tifffile

        output_file = f"{output_prefix}_demosaiced.tiff"
        # Permute to [4, 3, H, W] so TIFF is interpreted as 4 timepoints of 3-channel images
        demosaiced_transposed = np.transpose(demosaiced, (3, 2, 0, 1))
        tifffile.imwrite(
            output_file, demosaiced_transposed.astype(np.uint16), imagej=True, metadata={"axes": "TCYX"}
        )
        # print(f"Saved TIFF stack to {output_file} (4 polarisation angles × 3 color channels)")
    elif output_format.lower() == "png":
        import matplotlib.pyplot as plt

        angle_names = ["0deg", "45deg", "90deg", "135deg"]
        for i, angle in enumerate(angle_names):
            output_file = f"{output_prefix}_{angle}.png"

            # Normalise to 0-1 for PNG
            # HARDCODED: 4096 for 12-bit images
            img = demosaiced[:, :, :, i] / 4096

            plt.imsave(output_file, img)
            # print(f"Saved {angle} polarisation to {output_file}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Use 'tiff' or 'png'.")

    return demosaiced


def cli_demosaic():
    """Command line interface for de-mosaicing raw polarimetric images."""
    parser = argparse.ArgumentParser(description="De-mosaic raw polarimetric images into separate channels.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the raw image file or directory (with --all flag).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4096,
        help="Image width in pixels (default: 4096).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=3000,
        help="Image height in pixels (default: 3000).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["uint8", "uint16"],
        help="Data type (uint8 or uint16). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: input filename without extension).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="tiff",
        choices=["tiff", "png"],
        help="Output format: 'tiff' for single stack, 'png' for 4 separate images (default: tiff).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Recursively process all .raw files in the input directory and subdirectories.",
    )
    args = parser.parse_args()

    metadata = {
        "width": args.width,
        "height": args.height,
    }
    if args.dtype:
        metadata["dtype"] = args.dtype

    if args.all:
        # Process all raw files recursively
        if not os.path.isdir(args.input_file):
            raise ValueError(f"When using --all flag, input_file must be a directory. Got: {args.input_file}")

        # Find all .raw files recursively
        from glob import glob

        raw_files = glob(os.path.join(args.input_file, "**", "*.raw"), recursive=True)

        if len(raw_files) == 0:
            print(f"No .raw files found in {args.input_file}")
            return

        print(f"Found {len(raw_files)} .raw files to process")

        # Process each file
        from tqdm import tqdm

        for raw_file in tqdm(raw_files, desc="Processing raw files"):
            try:
                demosaic_raw_image(raw_file, metadata, args.output_prefix, args.format)
            except Exception as e:
                print(f"Error processing {raw_file}: {e}")
                continue
    else:
        # Process single file
        demosaic_raw_image(args.input_file, metadata, args.output_prefix, args.format)
