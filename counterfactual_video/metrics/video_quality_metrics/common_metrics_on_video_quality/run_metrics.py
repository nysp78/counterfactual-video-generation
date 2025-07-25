import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

# ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 1
VIDEO_LENGTH = 24
CHANNEL = 3
SIZE = 512
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
# device = torch.device("cpu")

import json
result = {}
only_final = False
# only_final = True
result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final)
# result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=only_final)
result['ssim'] = calculate_ssim(videos1, videos2, only_final=only_final)
result['psnr'] = calculate_psnr(videos1, videos2, only_final=only_final)
result['lpips'] = calculate_lpips(videos1, videos2, device, only_final=only_final)
print(json.dumps(result, indent=4))