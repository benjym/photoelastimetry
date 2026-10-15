# Quick image-to-retardation inversion

`image-to-retardation` converts polarisation intensities directly to optical
retardation and direction, independently per colour. It requires no material
coefficient, thickness, wavelength, JSON config, stress fitting or unwrapping.

```bash
image-to-retardation sample.tiff
image-to-retardation sample.tiff --input-stokes 0 0 1 --format png -o results/sample
image-to-retardation sample.tiff --bright-field bright.tiff --dark-field dark.tiff
```

Run from a source checkout without reinstalling with
`python -m photoelastimetry.quick sample.tiff`.

## Inputs and illumination

Use a TIFF produced by `demosaic-raw` or `stress-to-image`, or a NumPy array
with shape `[H, W, C, 4]`. The final axis must be ordered **0°, 45°, 90°, 135°**.
Monochrome arrays shaped `[H, W, 4]` are also accepted. TIFFs use the existing
package loader and its camera channel-order conventions. An ordinary RGB or
greyscale photograph does not contain the four analyser measurements needed.

Use `--raw-mosaic` for 2D TIFF/NPY sensor mosaics with the package's 4×4 pattern.
Raw `.raw` frames are demosaiced automatically and need the usual adjacent
`recordingMetadata.json`. Raw recording directories are also accepted.

`--input-stokes S1 S2 S3` specifies ideal, fully polarised illumination. The default
is circular `[0, 0, 1]`; use `[0, 0, -1]` for the opposite handedness
or `[1, 0, 0]` for linear light.
Other linear orientations are supported; mixed elliptical and partially polarised
states are rejected.

The command assumes the **lowest inverse branch**: retardation is in `[0, π]`
for linear light and `[0, π/2]` for circular light. A full `2π` period cannot be
uniquely recovered from these measurements. Higher retardations fold into this
range. Direction is relative to the analyser's 0° axis and follows the package's
Mueller model. Linear direction is returned in `[0, π/2)` (principal-axis ambiguity);
circular direction is in `[-π/2, π/2)`.

## Bright and dark fields

Both flags are optional and independent. References must match the input shape
and channel ordering. With both, correction is `(sample - dark) / (bright - dark)`;
with bright only it is `sample / bright`; with dark only it is `sample - dark`.
Correction happens before Stokes normalisation. `--raw-mosaic` applies to the
sample and references alike.

The bright reference must measure detector gain under uniform, unpolarised
illumination (equal expected intensity at every analyser angle). A polarised
blank can remove the physical polarisation signal when used as a flat field.

Nonpositive or nonfinite bright-minus-dark response, negative corrected samples,
and nonpositive total intensity produce invalid pixels. Linear isoclinics also produce invalid pixels. Zero polarisation signal leaves
direction undefined; on the lowest circular branch it still gives zero retardation. Values beyond the physical
retardation domain due to noise are clipped to the branch bounds.

## Outputs

By default, write `sample_retardation.tiff` and `sample_direction.tiff`. Both are
float32 maps **in radians**, with one channel per input colour and NaN for invalid
pixels. TIFFs use the package's ImageJ channel layout `[C, H, W]`.

`--format png` writes `sample_retardation_c0.png`, `sample_direction_c0.png`, etc.
These are 8-bit greyscale previews with fixed scales: black to white spans the
retardation branch or the direction interval stated above. Invalid pixels are
transparent. Use TIFF for quantitative analysis. `--format tif` is also supported.
`-o/--output-prefix` controls the destination prefix and creates parent directories.
