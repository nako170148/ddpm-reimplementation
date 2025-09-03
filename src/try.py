import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 設定 ---
T = 200  # 拡散ステップ数を少なめにして時間短縮
IMAGE_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 5  # 学習時間を抑えつつ精度向上

# --- ノイズスケジュール ---
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# --- 前向き拡散プロセス ---
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

# --- モデル定義（少し深めに強化） ---
class SimpleDDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    def forward(self, x, t):
        return self.net(x)

# --- データセット ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 学習処理 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleDDPM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    for step, (x, _) in enumerate(tqdm(dataloader)):
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device).long()
        noise = torch.randn_like(x)
        x_t = q_sample(x, t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- 逆拡散プロセス ---
@torch.no_grad()
def p_sample(model, x, t):
    betas_t = betas[t].to(x.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(x.device)
    sqrt_recip_alphas_t = (1.0 / torch.sqrt(alphas[t])).to(x.device)

    model_mean = sqrt_recip_alphas_t[:, None, None, None] * (x - betas_t[:, None, None, None] / sqrt_one_minus_alphas_cumprod_t[:, None, None, None] * model(x, t))

    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        posterior_variance_t = betas[t].to(x.device)
        return model_mean + torch.sqrt(posterior_variance_t)[:, None, None, None] * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t)
    return img

# --- 生成結果の表示 ---
model.eval()
generated_imgs = p_sample_loop(model, shape=(4, 1, IMAGE_SIZE, IMAGE_SIZE))

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axs[i].imshow(generated_imgs[i, 0].cpu(), cmap="gray")
    axs[i].axis("off")
    axs[i].set_title(f"Sample {i+1}")
plt.tight_layout()
plt.show()
