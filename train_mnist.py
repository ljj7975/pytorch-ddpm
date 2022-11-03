"""
Training a DDPM for MNIST dataset
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from datetime import datetime
from utils.positional_embedding import positional_encoding_1d
import os


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int, n_classes:int, n_T:int) -> None:
        super(DummyEpsModel, self).__init__()
        # class_bias is learnable while positional encoding is not
        self.class_bias = nn.Embedding(n_classes, 64)
        self.register_buffer("position_embedding", positional_encoding_1d(128, n_T))

        self.layer_1 = blk(n_channel, 64)
        self.layer_2 = blk(64, 128)
        self.layer_3 = blk(128, 256)
        self.layer_4 = blk(256, 512)
        self.layer_5 = blk(512, 256)
        self.layer_6 = blk(256, 128)
        self.layer_7 = blk(128, 64)
        self.layer_8 = nn.Conv2d(64, n_channel, 3, padding=1)

    def forward(self, x, t, y) -> torch.Tensor:
        class_embedding = self.class_bias(y)[:, :, None, None]
        positional_embedding = self.position_embedding[t][:, :, None, None]

        x_1 = self.layer_1(x) + class_embedding
        x_2 = self.layer_2(x_1) + positional_embedding
        x_3 = self.layer_3(x_2)
        x_4 = self.layer_4(x_3)
        x_5 = self.layer_5(x_4) + x_3
        x_6 = self.layer_6(x_5) + x_2 + positional_embedding
        x_7 = self.layer_7(x_6) + x_1 + class_embedding
        x_8 = self.layer_8(x_7)
        return x_8


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts, y))

    def sample(self, n_sample: int, yh:torch.Tensor, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T, yh)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


def train_mnist(n_epoch: int = 50, n_classes: int = 10, device="cuda:0") -> None:

    n_T = 500
    ddpm = DDPM(eps_model=DummyEpsModel(1, n_classes, n_T), betas=(1e-4, 0.02), n_T=n_T)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=3e-4)

    output_dir = f"./{datetime.now().strftime('%b-%H-%M-%S')}"
    os.makedirs(output_dir)

    yh = torch.randint(0,9, (16,))
    print(f"val sample labels: {yh}")
    yh = yh.to(device)

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = ddpm(x, y)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"epoch: {i}, loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, yh, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"{output_dir}/ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"{output_dir}/ddpm_mnist.pth")


if __name__ == "__main__":
    train_mnist()
