import json
import os
from glob import glob

import numpy as np
from tqdm import tqdm


def read_raw(filename, metadata):
    with open(filename, "rb") as f:
        # work out if it is 8bit or 16 bit
        eight_bit_file_size = metadata["width"] * metadata["height"] * 1
        actual_file_size = f.seek(0, 2)

        if actual_file_size == eight_bit_file_size:
            metadata["dtype"] = "uint8"
        elif actual_file_size == eight_bit_file_size * 2:
            metadata["dtype"] = "uint16"
        else:
            raise ValueError(
                f"File size does not match expected size for 8bit or 16bit data. Got {actual_file_size} bytes, expected {eight_bit_file_size} or {eight_bit_file_size * 2} bytes."
            )

        # f.seek(0)

        data = np.memmap(f, dtype=metadata["dtype"], mode="r", offset=0)
        data = data.reshape(
            (
                metadata["height"],
                metadata["width"],
            )
        )
        return data


def load_raw(foldername):
    json_file = foldername + "/recordingMetadata.json"
    if not os.path.exists(json_file):
        raise ValueError(f"Metadata file {json_file} does not exist. Please check the folder name.")

    frame_folder = foldername + "/0000000/"
    all_frames = glob(frame_folder + "frame*.raw")
    # frame_file = frame_folder + f"frame{str(0).zfill(10)}.raw"

    with open(json_file) as f:
        metadata = json.load(f)

    if len(all_frames) == 0:
        raise ValueError(f"No frames found in {frame_folder}.")
    else:
        # take median over all frames
        data = []
        for frame_file in tqdm(all_frames):
            data.append(read_raw(frame_file, metadata))
        data = np.median(np.array(data), axis=0)

    data = split_channels(data)

    return data, metadata


def split_channels(data):
    """
    Splits the data into its respective polarisation channels. Each superpixel
    is 4x4 pixels, and the channels are arranged in the following order:

    R_0 | R_45 | G1_0 | G1_45
    R_135 | R_90 | G1_135 | G1_90
    G2_0 | G2_45 | B_0 | B_45
    G2_135 | G2_90 | B_135 | B_90
    """

    # Reshape the data into a 4D array
    R_0 = data[0::4, 0::4]
    R_45 = data[0::4, 1::4]
    G1_0 = data[0::4, 2::4]
    G1_45 = data[0::4, 3::4]
    R_135 = data[1::4, 0::4]
    R_90 = data[1::4, 1::4]
    G1_135 = data[1::4, 2::4]
    G1_90 = data[1::4, 3::4]
    G2_0 = data[2::4, 0::4]
    G2_45 = data[2::4, 1::4]
    B_0 = data[2::4, 2::4]
    B_45 = data[2::4, 3::4]
    G2_135 = data[3::4, 0::4]
    G2_90 = data[3::4, 1::4]
    B_135 = data[3::4, 2::4]
    B_90 = data[3::4, 3::4]

    # Stack the channels into a 4D array
    I0 = np.stack((R_0, G1_0, G2_0, B_0), axis=-1)
    I90 = np.stack((R_90, G1_90, G2_90, B_90), axis=-1)
    I45 = np.stack((R_45, G1_45, G2_45, B_45), axis=-1)
    I135 = np.stack((R_135, G1_135, G2_135, B_135), axis=-1)

    # data is a 4D array with shape (height, width, colour, polarisation)
    data = np.stack(
        (
            I0,
            I45,
            I90,
            I135,
        ),
        axis=-1,
    )
    return data


def save_image(filename, data, metadata={}):
    """
    Save image data to a file in various formats.

    This function saves image data to disk in one of several supported formats,
    automatically determining the format from the file extension.

    Parameters
    ----------
    filename : str
        Path to the output file. The file extension determines the format.
        Supported extensions: .npy, .raw, .png, .jpg, .jpeg, .tiff, .tif
    data : numpy.ndarray
        Image data to save. The data will be cast to the appropriate dtype
        based on the file format (uint8 for .png/.jpg, uint16 for .tiff, etc.)
    metadata : dict
        Dictionary containing metadata about the image. For .raw format, must
        contain a "dtype" key specifying the data type to use when saving. Default is empty.

    Raises
    ------
    ValueError
        If the file extension is not one of the supported formats.

    Notes
    -----
    - .npy files preserve the original data type and shape
    - .raw files are saved as binary with dtype specified in metadata
    - .png and .jpg files convert data to uint8
    - .tiff/.tif files convert data to uint16
    - matplotlib is used for .png and .jpg formats
    - tifffile library is used for .tiff/.tif formats

    Examples
    --------
    >>> data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    >>> metadata = {"dtype": "uint8"}
    >>> save_image("output.png", data, metadata)
    """
    output_folder = os.path.dirname(filename)
    if output_folder != "" and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if filename.endswith(".npy"):
        np.save(filename, data)
    elif filename.endswith(".raw"):
        with open(filename, "wb") as f:
            data.astype(metadata["dtype"]).tofile(f)
    elif filename.endswith(".png"):
        import matplotlib.pyplot as plt

        plt.imsave(filename, data.astype(np.uint8))
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        import matplotlib.pyplot as plt

        plt.imsave(filename, data.astype(np.uint8))
    elif filename.endswith(".tiff") or filename.endswith(".tif"):
        import tifffile

        if data.ndim == 2:
            tifffile.imwrite(filename, data.astype("<f4"))
        elif data.ndim == 3:
            transposed = np.transpose(data, (2, 0, 1))
            tifffile.imwrite(filename, transposed.astype("<f4"), imagej=True, metadata={"axes": "CYX"})
        elif data.ndim == 4:
            # Permute to [4, 3, H, W] so TIFF is interpreted as 4 timepoints of 3-channel images
            transposed = np.transpose(data, (3, 2, 0, 1))
            tifffile.imwrite(filename, transposed.astype("<f4"), imagej=True, metadata={"axes": "TCYX"})
    else:
        raise ValueError(
            f"Unsupported file format for {filename}. Supported formats are .npy, .raw, .png, .jpg, .jpeg, .tiff, .tif"
        )


def load_image(filename, metadata=None):
    """
    Load image data from a file in various formats.

    This function loads image data from disk in one of several supported formats,
    automatically determining the format from the file extension.

    Parameters
    ----------
    filename : str
        Path to the input file. The file extension determines the format.
        Supported extensions: .npy, .raw, .png, .jpg, .jpeg, .tiff, .tif
    metadata : dict, optional
        Dictionary containing metadata about the image. For .raw format, must
        contain "width", "height", and "dtype" keys specifying the image dimensions
        and data type. Default is None.

    Returns
    -------
    numpy.ndarray
        Loaded image data.

    Raises
    ------
    ValueError
        If the file extension is not one of the supported formats.

    Notes
    -----
    - .npy files preserve the original data type and shape
    - .raw files are read as binary with dtype and shape specified in metadata
    - .png and .jpg files are loaded as uint8 arrays
    - .tiff/.tif files are loaded as uint16 arrays
    - matplotlib is used for .png and .jpg formats
    - tifffile library is used for .tiff/.tif formats

    Examples
    --------
    >>> metadata = {"width": 100, "height": 100, "dtype": "uint8"}
    >>> data = load_image("input.raw", metadata)
    """
    if filename.endswith(".npy"):
        data = np.load(filename)
    elif filename.endswith(".raw"):
        with open(filename, "rb") as f:
            data = np.memmap(
                f,
                dtype=metadata["dtype"],
                mode="r",
                offset=0,
                shape=(metadata["height"], metadata["width"]),
            )
            data = np.array(data)
    elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        import matplotlib.pyplot as plt

        data = plt.imread(filename)
    elif filename.endswith(".tiff") or filename.endswith(".tif"):
        import tifffile

        data = tifffile.imread(filename)

        # transpose data to XYC or XYCP format
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))
        elif data.ndim == 4:
            data = np.transpose(data, (2, 3, 1, 0))

    else:
        raise ValueError(
            f"Unsupported file format for {filename}. Supported formats are .npy, .raw, .png, .jpg, .jpeg, .tiff, .tif"
        )

    if metadata is None:
        metadata = {}
        metadata["dtype"] = str(data.dtype)
        metadata["height"] = data.shape[0]
        metadata["width"] = data.shape[1]

    return data, metadata


def bin_image(data, binning):
    """
    Bin the image by the specified factor.

    Parameters
    ----------
    data : numpy.ndarray
        Input image data to be binned.
    binning : int
        Binning factor. The image dimensions will be reduced by this factor.

    Returns
    -------
    numpy.ndarray
        Binned image data.
    """
    if binning <= 1:
        return data

    # Calculate new shape
    new_height = data.shape[0] // binning
    new_width = data.shape[1] // binning

    # Reshape and bin
    binned_data = (
        data[: new_height * binning, : new_width * binning]
        .reshape(new_height, binning, new_width, binning, *data.shape[2:])
        .mean(axis=(1, 3))
    )

    return binned_data
