import torch
from PIL import Image

def swap(regs, crop_size):
    reg_w, reg_h = regs[0].size
    k = 2

    Q_rd = torch.rand((crop_size, crop_size))
    Q_rd = Q_rd * (2.0*k) - k

    Q_row = torch.arange(0, crop_size).unsqueeze(0).repeat(crop_size, 1)
    Q_row_rd = Q_row + Q_rd
    
    Q_col = torch.arange(0, crop_size).unsqueeze(1).repeat(1, crop_size)
    Q_col_rd = Q_col + Q_rd

    _, sigma_row = torch.sort(input = Q_row_rd, dim = -1)
    _, sigma_col = torch.sort(input = Q_col_rd, dim = 0)

    swap_reg_id_row = (sigma_row + Q_col * crop_size).view(-1)
    swap_reg_id_col = (Q_row + sigma_col * crop_size).view(-1)

    sigma = torch.index_select(swap_reg_id_row, dim = 0, index = swap_reg_id_col)

    swap_regs = list(map(regs.__getitem__, sigma))

    swap_img = Image.new('RGB', (reg_w * crop_size, reg_h * crop_size))
    for id, reg in enumerate(swap_regs):
        row_id = int(id / crop_size)
        col_id = int(id % crop_size)
        swap_img.paste(reg, (col_id * crop_size, row_id * crop_size))

    sigma = ( sigma - (crop_size*crop_size)//2 ) / (crop_size*crop_size)
    
    return swap_img, sigma