import os
import sys

import json5
import matplotlib.pyplot as plt
import numpy as np

import photoelastimetry.io
from photoelastimetry.generate.disk import generate_synthetic_brazil_test
from photoelastimetry.image import compute_normalised_stokes, compute_stokes_components

ANGLE_LABELS = ["0", "45", "90", "135"]


def load_params(filename):
    with open(filename, "r") as f:
        return json5.load(f)


def prepare_experimental_data(params):
    data, _ = photoelastimetry.io.load_image(params["input_filename"])
    if "dark_field" in params:
        dark_field, _ = photoelastimetry.io.load_image(params["dark_field"])
        data = data - dark_field

    if params.get("crop") is not None:
        x1, x2, y1, y2 = params["crop"]
        data = data[y1:y2, x1:x2, :, :]

    if params.get("binning") is not None:
        data = photoelastimetry.io.bin_image(data, params["binning"])

    if data.ndim != 4 or data.shape[2] != 3 or data.shape[3] != 4:
        raise ValueError(f"Expected image data with shape [H, W, 3, 4]. Got {data.shape}.")

    return data.astype(float)


def parse_angle_order(order):
    if order is None:
        return [0, 1, 2, 3]

    parsed = []
    for angle in order:
        if isinstance(angle, str):
            parsed.append(ANGLE_LABELS.index(angle.replace("deg", "").replace("°", "")))
        else:
            parsed.append(int(angle))

    if sorted(parsed) != [0, 1, 2, 3]:
        raise ValueError(f"Angle order must be a permutation of [0, 1, 2, 3]. Got {order}.")
    return parsed


def normalised_stokes_from_images(images, angle_order=None):
    images = images[:, :, :, parse_angle_order(angle_order)]
    stokes = []
    for colour_index in range(images.shape[2]):
        s0, s1, s2 = compute_stokes_components(
            images[:, :, colour_index, 0],
            images[:, :, colour_index, 1],
            images[:, :, colour_index, 2],
            images[:, :, colour_index, 3],
        )
        s1_hat, s2_hat = compute_normalised_stokes(s0, s1, s2)
        stokes.append((s1_hat, s2_hat))
    return stokes


def generate_fake_data(params, shape):
    height, width = shape
    radius = params.get("disk_radius", params.get("radius", 0.01))
    load = params.get("load", 1.0)

    x = np.linspace(-radius, radius, width)
    y = np.linspace(radius, -radius, height)
    x_grid, y_grid = np.meshgrid(x, y)
    mask = np.sqrt(x_grid**2 + y_grid**2) <= radius

    wavelengths = np.asarray(params["wavelengths"], dtype=float) * 1e-9
    stress_optic_coefficients = np.asarray(params["C"], dtype=float)
    incoming_stokes = np.asarray(params.get("S_i_hat", [0.0, 0.0, 1.0]), dtype=float)

    synthetic_images, *_ = generate_synthetic_brazil_test(
        x_grid,
        y_grid,
        load,
        radius,
        incoming_stokes,
        mask,
        wavelengths=wavelengths,
        thickness=params["thickness"],
        C=stress_optic_coefficients,
        polarisation_efficiency=params.get("polarisation_efficiency", 1.0),
    )

    return synthetic_images


def plot_comparison(experimental_stokes, fake_stokes, output_filename):
    colour_names = ["red", "green", "blue"]
    column_specs = [
        ("experimental S1/S0", 0, experimental_stokes),
        ("fake S1/S0", 0, fake_stokes),
        ("experimental S2/S0", 1, experimental_stokes),
        ("fake S2/S0", 1, fake_stokes),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), layout="constrained")
    last_image = None

    for row, colour_name in enumerate(colour_names):
        for col, (title, component_index, stokes_source) in enumerate(column_specs):
            ax = axes[row, col]
            values = stokes_source[row][component_index]
            last_image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(colour_name)

    fig.colorbar(last_image, ax=axes, shrink=0.8, label="normalised Stokes component")
    fig.suptitle("Experimental vs fake disk normalised Stokes components")

    output_folder = os.path.dirname(output_filename)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    fig.savefig(output_filename, dpi=200)
    plt.close(fig)


def main():
    config_filename = sys.argv[1] if len(sys.argv) > 1 else "json/disk2.json5"
    params = load_params(config_filename)

    experimental_images = prepare_experimental_data(params)
    fake_images = generate_fake_data(params, experimental_images.shape[:2])

    experimental_angle_order = params.get("comparison_experimental_angle_order")
    fake_angle_order = params.get("comparison_fake_angle_order")

    experimental_stokes = normalised_stokes_from_images(experimental_images, experimental_angle_order)
    fake_stokes = normalised_stokes_from_images(fake_images, fake_angle_order)

    output_filename = params.get("comparison_output_filename", "images/disk2/stokes_comparison.png")
    plot_comparison(experimental_stokes, fake_stokes, output_filename)

    print(f"Saved {output_filename}")
    print(f"Experimental/fake image shape: {experimental_images.shape}")
    print(f"Experimental angle order: {parse_angle_order(experimental_angle_order)}")
    print(f"Fake angle order: {parse_angle_order(fake_angle_order)}")


if __name__ == "__main__":
    main()
