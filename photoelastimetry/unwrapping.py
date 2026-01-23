import maxflow
import numpy as np


def unwrap_angles_graph_cut(angles, quality=None):
    """
    Unwrap angles from the range [0, pi/2] to [0, pi] using a graph-cut method.

    This function formulates the unwrapping problem as a binary labeling problem (Markov Random Field),
    where for each pixel we decide whether to add 0 or pi/2 to the measured angle.
    The objective is to minimize the squared difference of the resulting angles between neighboring pixels.

    The energy function minimized is:
        E(x) = sum_{neighbors i,j} Q_ij * (theta_new_i - theta_new_j)^2
    where theta_new = theta_measured + x * pi/2, with x in {0, 1}.

    Parameters
    ----------
    angles : array_like
        Input angles in radians, assumed to be in the range [0, pi/2].
        Shape (H, W).
    quality : array_like, optional
        Quality map of the same shape as angles. Higher values indicate higher confidence.
        If None, a uniform quality of 1.0 is used.
        The smoothness terms are weighted by the minimum quality of the two neighbors.

    Returns
    -------
    unwrapped_angles : ndarray
        The unwrapped angles in the range [0, pi].
        Shape (H, W).
    """
    angles = np.array(angles, dtype=np.float64)
    H, W = angles.shape

    if quality is None:
        quality = np.ones((H, W), dtype=np.float64)
    else:
        quality = np.array(quality, dtype=np.float64)

    # The energy function decomposes into:
    # E(x) = Constant + sum_i U_i(x_i) + sum_{i,j} W_ij * (x_i != x_j)
    # Actually, for the squared difference metric:
    # Pairwise term W_ij is constant Q_ij * pi^2/4 (symmetric).
    # Unary term for node i comes from data differences with neighbors.
    # U_i is the "cost penalty" for setting x_i = 1 relative to x_i = 0.

    # Initialize capacities
    # We will accumulate the 'terminal' capacities diffs here.
    # Positive value means we add capacity to t-link to Sink (penalize 1).
    # Negative value means we add capacity to t-link to Source (penalize 0).
    unary_diff = np.zeros((H, W), dtype=np.float64)

    # Parameters
    PI = np.pi
    W_coeff = PI**2 / 4.0

    # --- Vertical Edges ---
    # Neighbors (y, x) and (y+1, x)
    # delta = theta(y) - theta(y+1)
    delta_v = angles[:-1, :] - angles[1:, :]

    # Weight Q_v = min(Q(y), Q(y+1))
    # Using geometric mean or min? Min is standard for "weakest link".
    q_v = np.minimum(quality[:-1, :], quality[1:, :])

    # Term added to Energy:
    # For pixel i=(y,x):   + Q * pi * delta * x_i
    # For pixel j=(y+1,x): - Q * pi * delta * x_j
    term_v = q_v * PI * delta_v

    unary_diff[:-1, :] += term_v
    unary_diff[1:, :] -= term_v

    # Pairwise weights
    weights_v = q_v * W_coeff

    # --- Horizontal Edges ---
    # Neighbors (y, x) and (y, x+1)
    delta_h = angles[:, :-1] - angles[:, 1:]
    q_h = np.minimum(quality[:, :-1], quality[:, 1:])

    term_h = q_h * PI * delta_h

    unary_diff[:, :-1] += term_h
    unary_diff[:, 1:] -= term_h

    weights_h = q_h * W_coeff

    # --- Graph Construction ---
    g = maxflow.Graph[float](H * W, H * W * 2)
    nodeids = g.add_grid_nodes((H, W))

    # Add n-links (edges between pixels)
    # Structure for add_grid_edges:
    # weights: array.
    # For structure=structure, usually defining neighborhood.
    # To add specific weights for vertical/horizontal, we might need separate calls or a constructed structure.
    # The default structure is von Neumann (4-connected).
    # add_grid_edges takes a single 'weights' array or one per direction.
    # If using 'weights' as array, it usually assumes isotropic or requires complex setup.
    # Easier way in PyMaxflow:
    # add_grid_edges(nodeids, weights, structure, symmetric)
    # If we pass a list of weight arrays?
    # Documentation says: "weights" can be a scalar or a numpy array with the same shape as nodeids.
    # If structure has multiple edges per node, how do we specify different weights?
    # We can call add_grid_edges multiple times with different structures!

    # Vertical edges: link (y,x) to (y+1,x). Structure: [0,0,0], [0,0,0], [0,1,0] ??
    # PyMaxflow structure is array shape (3,3). Center is (1,1).
    # Structure for down:
    # [[0,0,0],
    #  [0,0,0],
    #  [0,1,0]]
    # This connects (y,x) to (y+1, x).

    s_vert = np.zeros((3, 3))
    s_vert[2, 1] = 1  # Down

    # Note: weights array must match nodeids shape, but for edges at boundary, they are ignored/handled.
    # When using `add_grid_edges`, if we provide `weights` array of shape (H,W),
    # it uses `weights[y,x]` as capacity for the edge starting at (y,x) defined by structure.
    # So for structure "Down", weights[y,x] is cap for edge (y,x)->(y+1,x).
    # We computed `weights_v` of shape (H-1, W). We need to pad it to (H,W).
    # The last row won't have a down neighbor so value doesn't matter (or handled automatically).

    w_v_padded = np.zeros((H, W), dtype=np.float64)
    w_v_padded[:-1, :] = weights_v

    g.add_grid_edges(nodeids, w_v_padded, structure=s_vert, symmetric=True)

    # Horizontal edges: (y,x) to (y, x+1).
    s_horiz = np.zeros((3, 3))
    s_horiz[1, 2] = 1  # Right

    w_h_padded = np.zeros((H, W), dtype=np.float64)
    w_h_padded[:, :-1] = weights_h

    g.add_grid_edges(nodeids, w_h_padded, structure=s_horiz, symmetric=True)

    # --- Add t-links (Unary) ---
    # unary_diff[i]: Cost difference (Cost1 - Cost0).
    # If > 0: prefer 0. Add cap to Link to Sink (1).
    # If < 0: prefer 1. Add cap to Link to Source (0).

    # g.add_grid_tedges(nodeids, source_cap, sink_cap)
    # source_cap: cost if node is 1 (cut source link? No, cut source link means node is disconnected from Source -> node is Sink (1).
    # Wait, terminology:
    # Cap(S->i): If cut, $i$ becomes Sink.
    # Cap(i->T): If cut, $i$ becomes Source.
    # We want to minimize cost.
    # If we want x=0 (Source), we want high Cap(i->T) or low Cap(S->i)?
    # If we pick Source, we cut i->T. So we pay Cap(i->T).
    # If we pick Sink, we cut S->i. We pay Cap(S->i).
    # Cost(0) = Cap(i->T).
    # Cost(1) = Cap(S->i).
    # We have diff D = Cost(1) - Cost(0).
    # If D > 0 (Cost(1) > Cost(0)), we prefer 0.
    # We can set Cost(0)=0, Cost(1)=D.
    # So Cap(i->T) = 0, Cap(S->i) = D.

    # If D < 0 (Cost(0) > Cost(1)), we prefer 1.
    # Cost(0) = -D, Cost(1) = 0.
    # Cap(i->T) = -D, Cap(S->i) = 0.

    # Source Capacity array (cost of being 1).
    source_caps = np.maximum(unary_diff, 0)
    # Sink Capacity array (cost of being 0).
    sink_caps = np.maximum(-unary_diff, 0)

    g.add_grid_tedges(nodeids, source_caps, sink_caps)

    # Compute maxflow
    g.maxflow()

    # Get results
    # get_grid_segments returns boolean array where True means Source (0)??
    # Check PyMaxflow docs:
    # "False means the node is in the source segment, True means the node is in the sink segment." -> Standard confusion.
    # Actually usually 0/1.
    # segment: "returns the segment... 0 for source, 1 for sink".

    sgm = g.get_grid_segments(nodeids)
    # sgm is boolean. True if Sink (1). False if Source (0).

    # x_i corresponds to sgm.
    # 0 -> keep angle.
    # 1 -> add pi/2.

    unwrapped = angles.copy()
    unwrapped[sgm] += PI / 2.0

    return unwrapped
