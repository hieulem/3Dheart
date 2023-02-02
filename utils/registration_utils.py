import torch
import numpy as np


def find_center(contour: torch.Tensor) -> torch.Tensor:
    """
    Find the center of the contour in the 2D contour points.

    Args:
        contour: contour defined as (num_points, 2)

    :return:
        center: center of the contour
    """
    return contour.mean(dim=0)


def find_contour_width(contour: torch.Tensor) -> torch.Tensor:
    """
    Finds the width of the contour. Note that this width corresponds to the length between
    the max and min point on width axis.

    Args:
        contour: contour defined as (num_points, 2)

    :return:
        width: width of the contour
    """
    return torch.max(contour[:, ]) - torch.min(contour[:, 0])


def find_contour_height(contour: torch.Tensor) -> torch.Tensor:
    """
    Finds the height of the contour. Note that this height corresponds to the length between
    the max and min point on height axis.

    Args:
        contour: contour defined as (num_points, 2)

    :return:
        height: height of the contour
    """
    return torch.max(contour[:, 1]) - torch.min(contour[:, 1])


def check_order(contours):
    """
    Checks the order of the contours. Some slices / contours are in the reverse order.

    Args:
        contours: contour defined as (num_points, 2)

    :return:
        order: 0 if order is retained, 1 if order is reversed, -1 if order is undetermined
    """
    first_contour = contours[0]
    last_contour = contours[-1]

    first_contour_width = find_contour_width(first_contour)
    last_contour_width = find_contour_width(last_contour)

    first_contour_height = find_contour_height(first_contour)
    last_contour_height = find_contour_height(last_contour)

    if first_contour_width > last_contour_width and first_contour_height > last_contour_height:
        return 0
    elif first_contour_width < last_contour_width and first_contour_height < last_contour_height:
        return 1
    else:
        return -1


def reorder_data(order, data):
    """
    Reorders the data according to the order.

    Args:
        order: 0 if order is retained, 1 if order is reversed, -1 if order is undetermined
        data: The data to be reordered

    :return:
        reordered_data: The reordered data
    """
    if order < 0:
        raise Exception("Contour order is undetermined.")
    elif order == 0:
        return data
    elif order == 1:
        return data[::-1]


def shift_2d_replace(data: (np.ndarray, torch.Tensor),
                     dx: int,
                     dy: int,
                     constant: float = 0.0) -> (np.ndarray, torch.Tensor):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    Args:
        data: The 2d numpy array to be shifted
        dx: The shift in x
        dy: The shift in y
        constant: The constant to replace rolled values with
    :return:
        shifted_data: The shifted array with "constant" where roll occurs
    """
    if type(data) is torch.Tensor:
        shifted_data = torch.roll(data, dx, dims=1)
    else:
        shifted_data = np.roll(data, dx, axis=1)

    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    if type(data) is torch.Tensor:
        shifted_data = torch.roll(shifted_data, dy, dims=0)
    else:
        shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data