import torch

def deepfool_batch(
    model,
    images,
    num_classes=10,
    overshoot=0.02,
    max_iter=50
):
    device = images.device
    batch_size = images.size(0)

    adv_images = images.clone().detach()

    for i in range(batch_size):
        image = images[i:i+1].clone().detach()
        image.requires_grad = True

        output = model(image)
        label = output.argmax(1).item()

        r_tot = torch.zeros_like(image)

        for _ in range(max_iter):
            output = model(image)
            current_label = output.argmax(1).item()

            if current_label != label:
                break

            grads = []
            for k in range(num_classes):
                model.zero_grad()
                if image.grad is not None:
                    image.grad.zero_()

                output[0, k].backward(retain_graph=True)
                grads.append(image.grad.detach().clone())

            pert = float("inf")
            w = None

            for k in range(num_classes):
                if k == label:
                    continue

                w_k = grads[k] - grads[label]
                f_k = output[0, k] - output[0, label]
                pert_k = torch.abs(f_k) / torch.norm(w_k.flatten(), p=2)

                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert + 1e-4) * w / torch.norm(w.flatten(), p=2)
            r_tot = r_tot + r_i

            image = images[i:i+1] + (1 + overshoot) * r_tot
            image = image.detach()
            image.requires_grad = True

        adv_images[i] = image.detach()

    return adv_images
