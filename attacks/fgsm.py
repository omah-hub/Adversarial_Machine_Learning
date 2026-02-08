import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon):
    

    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Collect gradient of the input
    grad_sign = images.grad.sign()

    # Create adversarial image
    adv_images = images + epsilon * grad_sign

    # Keep pixel values in valid range [0,1]
    adv_images = torch.clamp(adv_images, 0, 1)
   # print("Max perturbation:", (adv_images - images).abs().max().item())
    return adv_images
