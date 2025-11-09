import numpy as np

def relative_pose_difference(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """
    Compute both the relative rotation angle (in degrees) and translation distance
    between two 4×4 transformation matrices.

    Parameters
    ----------
    T1, T2 : np.ndarray
        4×4 homogeneous transformation matrices representing SE(3) poses.

    Returns
    -------
    tuple[float, float]
        (rotation_angle_deg, translation_distance)
    """
    # Validate input
    if T1.shape != (4, 4) or T2.shape != (4, 4):
        raise ValueError("Both transformation matrices must be 4×4.")

    # Extract rotation and translation
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Relative rotation
    R_rel = R1.T @ R2
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    # Relative translation (in same frame as T1)
    t_rel = R1.T @ (t2 - t1)
    translation_diff = np.linalg.norm(t_rel)

    return theta_deg, translation_diff

# # Example usage
# if __name__ == "__main__":
#     T1 = np.array([
#                 [
#                     0.994202528598172,
#                     0.005002013557653551,
#                     -0.10740722503337437,
#                     -0.6358784951170593
#                 ],
#                 [
#                     -0.03668557424802477,
#                     -0.9231980633423114,
#                     -0.38256960736982026,
#                     4.332489637353011
#                 ],
#                 [
#                     -0.10107176050259309,
#                     0.38429196674062127,
#                     -0.9176623472321023,
#                     -3.3170780953424184
#                 ],
#                 [
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0
#                 ]
#             ])
#     T2 = np.array([
#                 [
#                     0.9995056495915787,
#                     -0.031281708372889606,
#                     0.0031481988167305513,
#                     0.6826417892972686
#                 ],
#                 [
#                     0.014919377374362736,
#                     0.3837787753594783,
#                     -0.9233045346809147,
#                     -3.8964960263723105
#                 ],
#                 [
#                     0.0276743313067819,
#                     0.9228950678732943,
#                     0.38405575777726586,
#                     6.725210396150199
#                 ],
#                 [
#                     0.0,
#                     0.0,
#                     0.0,
#                     1.0
#                 ]
#             ])

#     print(relative_pose_difference(T1,T2))