# attacks/pgd.py

import torch
import torch.nn.functional as F

def pgd_attack(
    model,
    images,
    labels,
    epsilon,
    alpha=0.01,
    steps=10
):
    """
    PGD Attack (L-infinity)
    """

    images = images.clone().detach()
    labels = labels.clone().detach()

    ori_images = images.clone().detach()

    for _ in range(steps):
        images.requires_grad = True

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Gradient ascent
        adv_images = images + alpha * images.grad.sign()

        # Projection
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images
