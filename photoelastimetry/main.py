import argparse
import os

import json5
import numpy as np
from scipy.ndimage import gaussian_filter

import photoelastimetry.io
import photoelastimetry.optimiser.equilibrium
import photoelastimetry.optimiser.intensity
import photoelastimetry.optimiser.stokes
import photoelastimetry.plotting
import photoelastimetry.seeding


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
        output_filename (str, optional): Path to save the output stress map image.
            If None, the stress map is not saved. Defaults to None. Can also be specified in params.

    Returns:
        numpy.ndarray: 2D array representing the stress map in Pascals.

    Notes:
        - Assumes incoming light is fully S1 polarised
        - Uses uniform stress-optic coefficient across all wavelengths
        - Assumes solid sample (NU = 1.0)
        - Wavelengths are automatically converted from nm to meters
    """

    if "folderName" in params:
        data, metadata = photoelastimetry.io.load_raw(params["folderName"])
    elif "input_filename" in params:
        data, metadata = photoelastimetry.io.load_image(params["input_filename"])
    else:
        raise ValueError("Either 'folderName' or 'input_filename' must be specified in params.")

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

    C = params["C"]  # Stress-optic coefficients in 1/Pa
    L = params["thickness"]  # Thickness in m
    WAVELENGTHS = np.array(params["wavelengths"]) * 1e-9  # Wavelengths in m
    NU = 1.0  # Solid sample
    if isinstance(C, list) or isinstance(C, np.ndarray):
        C_VALUES = C
    else:
        C_VALUES = [
            C,
            C,
            C,
        ]  # Stress-optic coefficients in 1/Pa

    # Get incoming polarisation state from config
    S_I_HAT = np.array(params["S_i_hat"])
    # Ensure it's 3 elements for consistency
    if len(S_I_HAT) == 2:
        S_I_HAT = np.append(S_I_HAT, 0.0)  # Add S3_hat = 0 for backward compatibility

    # Calculate stress map from image
    n_jobs = params.get("n_jobs", -1)  # Default to using all cores

    # Phase Decomposed Seeding
    seeding_config = params.get("seeding", {})
    use_seeding = seeding_config.get("enabled", True)
    n_max = seeding_config.get("n_max", 6)
    sigma_max = seeding_config.get("sigma_max", 10e6)

    initial_stress_map = None
    if use_seeding and params.get("solver") != "stokes":
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
            n_jobs=n_jobs,
        )

    if params.get("solver") == "stokes":
        stress_map = photoelastimetry.optimiser.stokes.recover_stress_map_stokes(
            data,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            initial_guess_map=initial_stress_map,
            n_jobs=n_jobs,
        )
    elif params.get("solver") == "intensity":
        stress_map, success_map = photoelastimetry.optimiser.intensity.recover_stress_map_intensity(
            data,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            initial_guess_map=initial_stress_map,
            n_jobs=n_jobs,
        )
    elif params.get("solver") == "global":
        # Global solver specific logic
        gs_params = params.get("global_solver", {})

        boundary_mask = None
        if "boundary_mask_file" in gs_params:
            import tifffile

            if os.path.exists(gs_params["boundary_mask_file"]):
                boundary_mask = tifffile.imread(gs_params["boundary_mask_file"]) > 0

        stress_map = photoelastimetry.optimiser.equilibrium.recover_stress_global(
            data,
            WAVELENGTHS,
            C_VALUES,
            NU,
            L,
            S_I_HAT,
            boundary_mask=boundary_mask,
            initial_stress_map=initial_stress_map,
            **gs_params,
        )
    else:
        raise ValueError("Solver type not recognized. Use 'stokes', 'intensity', or 'global'.")

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

    with open(params["p_filename"]) as f:
        dict, p = photoelastimetry.io.load_file(f)

    sigma = photoelastimetry.io.load_image(params["stress_filename"], dict)

    sigma_xx = sigma[:, :, 2]
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]

    if params["scattering"]:
        # Add scattering
        sigma_xx = gaussian_filter(sigma_xx, sigma=params["scattering"])
        sigma_xy = gaussian_filter(sigma_xy, sigma=params["scattering"])
        sigma_yy = gaussian_filter(sigma_yy, sigma=params["scattering"])

    # Compute principal stresses
    sigma_avg = (sigma_xx + sigma_yy) / 2
    R = np.sqrt(((sigma_xx - sigma_yy) / 2) ** 2 + sigma_xy**2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R

    # Stress difference and retardation
    delta_sigma = sigma_1 - sigma_2

    # Retardation
    delta = (2 * np.pi * params["t"] / params["lambda_light"]) * params["C"] * delta_sigma

    # Fringe order
    # N = delta / (2 * np.pi)

    # Visualize Isochromatic Fringe Pattern
    fringe_intensity = np.sin(delta / 2) ** 2  # Fringe pattern

    # Isoclinic angle (principal stress orientation)
    phi = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy)  # Angle in radians

    # Plot the results
    if "output_filename" in params:
        output_filename = params["output_filename"]
    else:
        output_filename = "output.png"
    photoelastimetry.plotting.plot_fringe_pattern(fringe_intensity, phi, filename=output_filename)


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
