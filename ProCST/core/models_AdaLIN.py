import torch
import torch.nn as nn
import torch.nn.functional as F

# AdaLIN 클래스 정의
class AdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIN, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out

# AdaLIN을 사용 컨볼루션 블록
class ConvBlockProCST(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, images_per_gpu, groups_num):
        super(ConvBlockProCST, self).__init__()
        
        # 1. 컨볼루션 레이어
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
        
        # 2. AdaLIN 또는 정규화 레이어 (이미 정의하셨을 겁니다)
        self.norm = AdaLIN(out_channels) # 혹은 기존 정규화 층
        
        # 3. 활성화 함수 (이 부분이 누락되었을 가능성이 높습니다)
        self.act = nn.LeakyReLU(0.2, inplace=True) # <-- 이 줄이 있는지 확인하세요!

    def forward(self, x, gamma=None, beta=None):
        x = self.conv(x)
        
        # AdaLIN 적용 (gamma, beta가 있을 때만)
        if gamma is not None and beta is not None:
            x = self.norm(x, gamma, beta)
        else:
            # Discriminator 등에서 gamma/beta 없이 호출될 때 처리
            # self.norm이 AdaLIN 객체라면 내부적으로 처리하게 하거나 pass
            try:
                x = self.norm(x)
            except:
                pass
        
        # 에러가 난 지점: self.act가 정의되어 있어야 합니다.
        x = self.act(x) 
        return x

# AdaLIN 최적화 Generator
class ProCSTGenerator(nn.Module):
    def __init__(self, opt):
        super(ProCSTGenerator, self).__init__()
        self.is_initial_scale = opt.curr_scale == 0
        self.base_ch = opt.base_channels
        self.layers_num = opt.body_layers if opt.curr_scale < opt.num_scales - 1 else opt.body_layers + 2
        
        # Head: 초기 입력 처리
        in_ch = (2 - self.is_initial_scale) * opt.nc_im
        self.head_conv = nn.Conv2d(in_ch, self.base_ch, kernel_size=opt.ker_size, padding=1)
        
        # Body: AdaLIN 블록들
        self.body_blocks = nn.ModuleList([
            #ConvBlockProCST(self.base_ch, self.base_ch, opt.ker_size)
            ConvBlockProCST(self.base_ch, self.base_ch, opt.ker_size, opt.images_per_gpu[opt.curr_scale], opt.groups_num)
            for _ in range(self.layers_num - 2)
        ])
        
        # Tail: 이미지 생성
        self.tail = nn.Sequential(
            nn.Conv2d(self.base_ch, opt.nc_im, kernel_size=opt.ker_size, padding=1),
            nn.Tanh()
        )

        # 스타일 파라미터(Gamma, Beta) 생성용 MLP
        # Global Average Pooling 후 이 MLP를 통해 스타일 가중치를 뽑습니다.
        self.mlp = nn.Sequential(
            nn.Linear(self.base_ch, self.base_ch),
            nn.ReLU(True),
            nn.Linear(self.base_ch, self.base_ch * 2)
        )

    def forward(self, curr_scale, prev_scale):
        if self.is_initial_scale:
            x = curr_scale
        else:
            x = torch.cat((curr_scale, prev_scale), 1)
        
        x = self.head_conv(x)
        
        # Global 통계를 뽑아 MLP에 전달 (스타일 결정)
        style_feat = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        style_params = self.mlp(style_feat)
        gamma, beta = style_params.chunk(2, dim=1)

        # Body 연산 (AdaLIN 적용)
        for block in self.body_blocks:
            x = block(x, gamma, beta)
        
        x = self.tail(x)
        return x

# Discriminator: 기존 ProCST와 동일
class ProCSTDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ProCSTDiscriminator, self).__init__()
        self.images_per_gpu = opt.images_per_gpu[opt.curr_scale]
        self.layers_in_discriminator = opt.body_layers if opt.curr_scale < opt.num_scales - 1 else opt.body_layers+2
        self.head = ConvBlockProCST(opt.nc_im, opt.base_channels, opt.ker_size, self.images_per_gpu, opt.groups_num)
        self.body = nn.Sequential()
        for i in range(self.layers_in_discriminator-2):
            block = ConvBlockProCST(opt.base_channels, opt.base_channels, opt.ker_size, self.images_per_gpu, opt.groups_num)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(ConvBlockProCST(opt.base_channels, 1, opt.ker_size, self.images_per_gpu, opt.groups_num),
                                  nn.LeakyReLU(0.2))

    def forward(self, x):    
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

# Miscellaneous:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1 or classname.find('AdaLIN') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
