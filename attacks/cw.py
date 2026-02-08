import torch
import torch.nn.functional as F

def cw_attack(
    model,
    images,
    labels,
    c=0.1,
    kappa=0,
    lr=0.01,
    max_iter=500
):
    device = images.device
    batch_size = images.size(0)

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    # Initialize w in arctanh space
    w = torch.atanh((images - 0.5) * 1.999999)
    w = w.detach()
    w.requires_grad = True

    optimizer = torch.optim.Adam([w], lr=lr)

    def f(x):
        logits = model(x)
        one_hot = F.one_hot(labels, logits.size(1)).float()
        real = torch.sum(one_hot * logits, dim=1)
        other = torch.max((1 - one_hot) * logits - one_hot * 1e4, dim=1)[0]
        return torch.clamp(real - other + kappa, min=0)

    for _ in range(max_iter):
        adv_images = 0.5 * (torch.tanh(w) + 1)

        l2_loss = F.mse_loss(adv_images, images, reduction="sum")
        f_loss = torch.sum(f(adv_images))
        loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    adv_images = 0.5 * (torch.tanh(w) + 1)
    return adv_images.detach()
