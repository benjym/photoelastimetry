import numpy as np
from scipy.interpolate import BSpline


class BSplineAiry:
    """
    Manages a tensor-product B-spline surface for the Airy stress function.

    This class pre-computes basis functions to allow fast evaluation of
    stress fields (derivatives of the Airy function) from a set of control points.

    The Airy stress function phi(x,y) is represented as:
        phi(x,y) = sum_i sum_j C_ij * B_i(x) * B_j(y)

    Stresses are:
        sigma_xx = d^2(phi)/dy^2
        sigma_yy = d^2(phi)/dx^2
        sigma_xy = -d^2(phi)/dxdy
    """

    def __init__(self, shape, knot_spacing, degree=3):
        """
        Initialize the B-spline basis for a given image shape.

        Parameters
        ----------
        shape : tuple
            Image shape (height, width).
        knot_spacing : int
            Approximate spacing between knots in pixels.
        degree : int
            Degree of the B-spline (default: 3 for cubic).
        """
        self.ny, self.nx = shape
        self.degree = degree
        self.knot_spacing = knot_spacing

        # Generate knots
        # We need knots to cover the range [0, n] with sufficient padding for the degree
        # interior knots
        # tx = np.arange(0, self.nx + knot_spacing, knot_spacing)
        # ty = np.arange(0, self.ny + knot_spacing, knot_spacing)

        # Add simpler padding
        # Standard way for clamped B-spline is repeating start/end knots degree+1 times
        # But for general coverage we just need enough support.
        # Let's use the standard "clamped" knot vector construction for the domain [0, L]

        def make_knots(length, step):
            # Interior knots
            interior = np.arange(0, length + step, step)
            if interior[-1] < length:
                interior = np.append(interior, length)

            # Pad with clamped ends
            t = np.concatenate(([interior[0]] * degree, interior, [interior[-1]] * degree))
            return t

        self.tx = make_knots(self.nx, knot_spacing)
        self.ty = make_knots(self.ny, knot_spacing)

        # Number of coefficients (control points)
        self.n_coeffs_x = len(self.tx) - degree - 1
        self.n_coeffs_y = len(self.ty) - degree - 1
        self.n_coeffs = self.n_coeffs_x * self.n_coeffs_y

        # Pre-evaluate basis functions on the pixel grid
        # We evaluate at pixel centers
        x_grid = np.arange(self.nx)
        y_grid = np.arange(self.ny)

        # Evaluate basis functions B(x) and derivatives
        # This creates matrices of shape (width, n_coeffs_x)
        self.Bx, self.dBx, self.ddBx = self._precompute_basis(x_grid, self.tx, self.n_coeffs_x)
        self.By, self.dBy, self.ddBy = self._precompute_basis(y_grid, self.ty, self.n_coeffs_y)

    def _precompute_basis(self, coords, knots, n_coeffs):
        """
        Compute B-spline basis matrix and its 1st and 2nd derivatives.
        Returns matrices of shape (len(coords), n_coeffs).
        """
        # We use scipy BSpline.design_matrix-like logic but explicit
        # We want to know the value of the i-th basis function at each coordinate.
        # B_mat[k, i] = B_i(coords[k])

        B_mat = np.zeros((len(coords), n_coeffs))
        dB_mat = np.zeros((len(coords), n_coeffs))
        ddB_mat = np.zeros((len(coords), n_coeffs))

        # Iterate over each basis function
        # This might be slow for very large grids, but it's done once.
        # A faster way is to realize only degree+1 functions are non-zero at any point.
        # But optimizing this pre-calc is secondary to the main loop speed.

        for i in range(n_coeffs):
            # Create a localized BSpline for the i-th basis function
            # The coefficient vector is 1 at i and 0 elsewhere
            c = np.zeros(n_coeffs)
            c[i] = 1.0
            spl = BSpline(knots, c, self.degree)

            B_mat[:, i] = spl(coords)
            dB_mat[:, i] = spl(coords, nu=1)
            ddB_mat[:, i] = spl(coords, nu=2)

        return B_mat, dB_mat, ddB_mat

    def get_stress_fields(self, coeffs_flat):
        """
        Compute stress fields from flat coefficient array.

        Parameters
        ----------
        coeffs_flat : array-like
            Flattened array of coefficients of length n_coeffs_x * n_coeffs_y.

        Returns
        -------
        sigma_xx, sigma_yy, sigma_xy : ndarray
            Stress fields of shape (height, width).
        """
        C = coeffs_flat.reshape(self.n_coeffs_y, self.n_coeffs_x)

        # sigma_xx = d^2(phi)/dy^2 = By'' * C * Bx.T
        # Shape: (ny, n_cy) @ (n_cy, n_cx) @ (n_cx, nx) -> (ny, nx)
        sigma_xx = self.ddBy @ C @ self.Bx.T

        # sigma_yy = d^2(phi)/dx^2 = By * C * Bx''.T
        sigma_yy = self.By @ C @ self.ddBx.T

        # sigma_xy = -d^2(phi)/dxdy = -(By' * C * Bx'.T)
        # Note: In image coordinates (y down), d/dy_img = -d/y_phy.
        # The cross derivative term gains a negative sign from the coordinate flip,
        # cancelling the negative sign in the Airey definition.
        # So sigma_xy = + d^2(phi)/dx_img dy_img
        sigma_xy = self.dBy @ C @ self.dBx.T

        return sigma_xx, sigma_yy, sigma_xy

    def project_stress_gradients(self, grad_s_xx, grad_s_yy, grad_s_xy):
        """
        Project stress field gradients back to B-spline coefficients.
        This effectively computes d(Loss)/d(Coeffs) given d(Loss)/d(Stress).

        Parameters
        ----------
        grad_s_xx, grad_s_yy, grad_s_xy : ndarray
            Gradients of the loss with respect to stress fields. (ny, nx)

        Returns
        -------
        grad_coeffs : ndarray
            Gradient with respect to flattened coefficients.
        """
        # The stress field generation is linear: S = A @ C @ B.T
        # The gradient backprop is: G_C = A.T @ G_S @ B

        # Contributions from sigma_xx = ddBy @ C @ Bx.T
        grad_C = self.ddBy.T @ grad_s_xx @ self.Bx

        # Contributions from sigma_yy = By @ C @ ddBx.T
        grad_C += self.By.T @ grad_s_yy @ self.ddBx

        # Contributions from sigma_xy = dBy @ C @ dBx.T
        grad_C += self.dBy.T @ grad_s_xy @ self.dBx

        return grad_C.flatten()

    def fit_stress_field(self, stress_map):
        """
        Fit B-spline coefficients to an initial stress map.

        Parameters
        ----------
        stress_map : ndarray
            Target stress map [H, W, 3] usually from seeding.

        Returns
        -------
        coeffs_flat : ndarray
            Flattened coefficients fitting the stress map.
        """
        target_s_xx = stress_map[:, :, 0]
        target_s_yy = stress_map[:, :, 1]
        target_s_xy = stress_map[:, :, 2]

        def loss_and_grad(coeffs):
            s_xx, s_yy, s_xy = self.get_stress_fields(coeffs)

            diff_xx = s_xx - target_s_xx
            diff_yy = s_yy - target_s_yy
            diff_xy = s_xy - target_s_xy

            # Handle NaNs in target (e.g. from failed seeding or masking)
            if np.isnan(stress_map).any():
                mask_xx = np.isnan(target_s_xx)
                mask_yy = np.isnan(target_s_yy)
                mask_xy = np.isnan(target_s_xy)

                diff_xx[mask_xx] = 0
                diff_yy[mask_yy] = 0
                diff_xy[mask_xy] = 0

            loss = np.sum(diff_xx**2 + diff_yy**2 + diff_xy**2)

            # Gradient
            # d(Loss)/ds = 2 * diff
            grad = self.project_stress_gradients(2 * diff_xx, 2 * diff_yy, 2 * diff_xy)

            return loss, grad

        from scipy.optimize import minimize

        # Start from zeros
        res = minimize(
            loss_and_grad,
            np.zeros(self.n_coeffs),
            jac=True,
            method="L-BFGS-B",
            options={"disp": False, "maxiter": 100},  # Quick fit
        )
        return res.x
