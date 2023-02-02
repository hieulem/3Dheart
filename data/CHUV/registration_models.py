import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from utils.registration_utils import find_center
from typing import Callable, List, Tuple, Union
from kornia.filters import gaussian_blur2d
from IPython import embed 

def get_loss(loss_type: str) -> Callable:
    """
    Returns the loss function based on the loss type.

    Args:
        loss_type: the loss type
    :return:
        loss function -> must be Callable
    """
    if loss_type == "mi":
        return mi_loss
    elif loss_type == "ncc":
        return ncc_loss
    elif loss_type == "chamfer":
        return chamfer_symmetric
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))


def chamfer_symmetric(a, b):
    N1 = a.shape[1]
    N2 = b.shape[1]

    y1 = a[:, :, None].repeat(1, 1, N2, 1)
    y2 = b[:, None].repeat(1, N1, 1, 1)

    diff = torch.sum((y1 - y2) ** 2, dim=3)

    loss1, _ = torch.min(diff, dim=1)
    loss2, _ = torch.min(diff, dim=2)

    loss = torch.sum(loss1) + torch.sum(loss2)
    return loss


def ncc_loss(roi, target, normalized=True):
    """
    Computes the normalized cross correlation value given a region of interest and target

    Args:
        roi: fixed_slice with padding
        target: current_slice
    :return:
        normalized cross correlation value
    """
    # Normalised Cross Correlation Equation
    cor = roi * target
    sum_cor = torch.sum(cor)

    if normalized:
        nor = torch.sqrt((torch.sum(torch.square(roi)))) * torch.sqrt(torch.sum(torch.square(target)))
        cor_metric = sum_cor / nor
    else:
        cor_metric = sum_cor

    return cor_metric


def mi_loss(x, y, normalized=True):
    """
    Computes the mutual information loss between two images.

    Args:
        x: (torch.Tensor) the first image
        y: (torch.Tensor) the second image
        normalized: (bool) normalize mutual information
    :return:
        mutual information
    """
    EPS = torch.finfo(float).eps

    x = torch.ravel(x)  # Image 1
    y = torch.ravel(y)  # Image 2

    bins = (256, 256)

    jh = torch.histogramdd(input=torch.stack([x, y], dim=1), bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    jh = gaussian_blur2d(jh.unsqueeze(0).unsqueeze(0), kernel_size=(3, 3), sigma=(1.0, 1.0),
                         border_type="reflect").squeeze()

    # compute marginal histograms
    jh = jh + EPS
    sh = torch.sum(jh)
    jh = jh / sh
    s1 = torch.sum(jh, dim=0).reshape((-1, jh.shape[0]))
    s2 = torch.sum(jh, dim=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((torch.sum(s1 * torch.log(s1)) + torch.sum(s2 * torch.log(s2)))
              / torch.sum(jh * torch.log(jh))) - 1
    else:
        mi = (torch.sum(jh * torch.log(jh)) - torch.sum(s1 * torch.log(s1))
              - torch.sum(s2 * torch.log(s2)))

    return mi


class ContourWarper(nn.Module):
    def __init__(self, num_src_contours: int) -> None:
        super().__init__()
        """
        Contour warper module. It shifts the source contours to the destination contours.
        
        Args:
            num_src_contours: number of source contours
        
        """
        self.num_src_contours = num_src_contours

    def forward(self, src_contours: List[torch.Tensor], src_shifts_dst: torch.Tensor) -> List[torch.Tensor]:
        """
        Shift each source contour to destination based on the model parameters (shift).
        Args:
            src_contours: source contours
            src_shifts_dst: shift of source contours to destination contours
        :return:
            shifted_src_contour: shifted source contours

        """
        shifted_src_contour = []
        for index in range(self.num_src_contours):
            shifted_src_contour.append(src_contours[index] + src_shifts_dst[index])
        return shifted_src_contour


class ContourModel(nn.Module):
    def __init__(self,
                 registration_type: str,
                 num_contours: int,
                 shift: bool = True,
                 ) -> None:
        super().__init__()
        """
        Contour model module. It defines a shift parameters based on registration type and num contours.
        Note: This is where you can also add other registration models such as rotation!
        
        Args:
            registration_type: registration type
            num_contours: number of contours
            shift: Enable shift
            
        """

        if num_contours <= 0:
            raise ValueError("num_contours must be positive")

        self.num_contours = num_contours

        if registration_type == "analytical" or registration_type == "anchor_optimization":
            self.num_parameters = self.num_contours - 1
        else:
            self.num_parameters = self.num_contours

        if shift:
            self.shift = nn.Parameter(torch.zeros(self.num_parameters, 2))
        else:
            self.register_buffer('shift', torch.zeros(self.num_parameters, 2))

        self.reset_model()

    def reset_model(self):
        """
        Reset the model.
        """
        torch.nn.init.zeros_(self.shift)

    def forward(self):
        """
        Return the shift.
        """
        return self.shift

    def forward_inverse(self):
        """
        Return the inverse shift.
        """
        return -1 * self.forward()


class OptimizationContourRegistrator(nn.Module):
    """Module, which performs optimization-based contour registration.
    Args:
        registration_type: Type of registration, either "analytical" or "anchor_optimization" or "free_optimization"
        num_contours: Number of contours in the stack
        optimizer: optimizer class used for the optimization.
        loss_fn: torch loss function.
        lr: learning rate for optimization.
        num_iterations: maximum number of iterations.
        tolerance: stop optimizing if loss difference is less. default 1e-4.
    """

    def __init__(self,
                 registration_type: str,
                 num_contours: int,
                 optimizer=torch.optim.Adam,
                 loss_fn: Callable = chamfer_symmetric,
                 lr: float = 1e-1,
                 num_iterations: int = 1000,
                 tolerance: float = 1e-4) -> None:

        super().__init__()
        self.warper = ContourWarper
        self.registration_type = registration_type
        self.model = ContourModel(registration_type=registration_type,
                                  num_contours=num_contours,
                                  shift=True)
        self.optimizer = optimizer
        self.lr = lr
        self.loss_fn = loss_fn
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.num_contours = num_contours

    def reset_model(self) -> None:
        """Calls model reset function."""
        self.model.reset_model()

    def register(self,
                 contours: list,
                 milestones: list, 
                 verbose: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Estimate the transformation' which warps src_img into dst_img by gradient descent.
        The shape of the tensors is not checked, because it may depend on the model, e.g. volume registration

        Args:
            contours: List of contours in the given stack
            verbose: if True, outputs loss every 10 iterations.
        :returns:
            the transformation between two contours, shape depends on the model,
        """
        self.reset_model()
        opt: torch.optim.Optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=0.1)

        prev_loss = 1e10

        for iteration in range(self.num_iterations):
            # compute gradient and update optimizer parameters
            opt.zero_grad()
            loss = torch.Tensor([0]).to(contours[0].device)

            if self.registration_type == "free_optimization":
                # We iterate through all possible pairs.
                for i in range(len(contours)):
                    dest_contour = contours[i]
                    # We break up the list of contours into two parts -> before and after the current contour.
                    source_contour = contours[:i] + contours[i + 1:]
                    loss += self.compute_loss(contour_dst=dest_contour, contour_src=source_contour,
                                              transform_model=self.model())

            elif self.registration_type == "anchor_optimization":
                # We fix the destination and source contours.
                dest_contour = contours[0]
                source_contour = contours[1:] 
                loss += self.compute_loss(contour_dst=dest_contour, contour_src=source_contour,
                                          transform_model=self.model())
            elif self.registration_type == "moving_anchor_optimization": 
                for i in range(len(contours)-1):
                    dest_contour = contours[i]
                    source_contour = [contours[i+1]]
                    loss += self.compute_loss(contour_dst=dest_contour, contour_src=source_contour,
                                            transform_model=self.model())
            current_loss = loss.item()
            if abs(current_loss - prev_loss) < self.tolerance: 
                # print(f"Loss = {current_loss:.4f}, iteration={iteration}, lr={lr}")
                break
            prev_loss = current_loss
            loss.backward() 
            if verbose and (iteration % 10000 == 0):
                lr = opt.param_groups[0]["lr"]
                print(f"Loss = {current_loss:.4f}, iteration={iteration}, lr={lr}")
            opt.step()
            scheduler.step()
        print(f"Loss = {current_loss:.4f}, iteration={iteration}, lr={lr}")
        return self.model()

    def get_center(self, cont1):
        A = cont1[:, None]
        B = cont1[:, :, None] 
        dist = torch.sum((A-B)**2,dim=3)
        _, opp_idx = torch.sort(dist,dim=2)
        opp_idx = opp_idx[:, :, -1].squeeze()
        cont1 = 0.5 * (cont1 + cont1[:, opp_idx])
        center1 = torch.mean(cont1, dim=1, keepdim=True)
        return center1

    def compute_loss(self,
                     contour_dst: torch.Tensor,
                     contour_src: torch.Tensor,
                     transform_model: torch.Tensor) -> torch.Tensor:
        """
        Warp img_src into img_dst with transform_model and returns loss.

        Args:
            contour_dst: contour in the destination slice
            contour_src: contour in the source slice
            transform_model: transformation model
        :returns:
            loss: loss between contour_dst and contour_src

        """
        loss = torch.Tensor([0]).to(contour_dst.device)
        warper = self.warper(len(contour_src))
        contour_src_to_dst = warper(contour_src, transform_model)

        for src_index in range(len(contour_src)):
            # cont1 = contour_dst.unsqueeze(0) 
            # cont2 = contour_src_to_dst[src_index].unsqueeze(0)
            # loss += self.loss_fn(cont1, cont2 )
 
            # center1 = self.get_center(cont1)
            # center2 = self.get_center(cont2)
            # center1 = torch.mean(cont1, dim=1, keepdim=True)
            # center2 = torch.mean(cont2, dim=1, keepdim=True)  
            # loss += torch.sum((center1 - center2) ** 2)

            center1 = contour_dst.unsqueeze(0) 
            center2 = contour_src_to_dst[src_index].unsqueeze(0)
            loss += torch.sum((center1 - center2) ** 2)


        return loss


class AnalyticalContourRegistrator:
    def __init__(self,
                 registration_type: str,
                 num_contours: int,
                 ) -> None:
        super().__init__()
        """
        Analytical contour registration class. It defines a contour model of shape (num_contours-1, 2), and finds
        the analytical solution for the contour registration based on aligning the centers of each contour.
        
        Args:
            registration_type: type of registration to perform (anchor_optimization, free_optimization, analytical)
            num_contours: number of contours in the stack
        """

        self.model = ContourModel(registration_type=registration_type,
                                  num_contours=num_contours,
                                  shift=True)
        # Turn off gradient because we are not doing optimization
        self.model.shift.requires_grad = False
        self.num_contours = num_contours

    def register(self,
                 contours: list,
                 verbose: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Fix the first center point
        fixed_point = find_center(contours[0])

        # Find the rest and compute alignment shift
        for i in range(1, self.num_contours):
            contour_center = find_center(contour=contours[i])
            self.model.shift[i - 1] = fixed_point - contour_center

        return self.model()