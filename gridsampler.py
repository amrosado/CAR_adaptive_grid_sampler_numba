import numba.cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function, gradcheck

from numba import cuda

BLOCK_SIZE = 256

class GridSamplerFunction(Function):
    @staticmethod
    def forward(ctx, img, kernels, offsets_h, offsets_v, offset_unit, padding, downscale_factor):
        assert isinstance(downscale_factor, int)
        assert isinstance(padding, int)

        ctx.padding = padding
        ctx.offset_unit = offset_unit

        b, c, h, w = img.size()
        assert h // downscale_factor == kernels.size(2)
        assert w // downscale_factor == kernels.size(3)

        img = nn.ReflectionPad2d(padding)(img)
        ctx.save_for_backward(img, kernels, offsets_h, offsets_v)

        output = img.new(b, c, h // downscale_factor, w // downscale_factor).zero_()

        with torch.no_grad():
            img = numba.cuda.as_cuda_array(img.detach())
            kernels = numba.cuda.as_cuda_array(kernels.detach())
            offsets_h = numba.cuda.as_cuda_array(offsets_h.detach())
            offsets_v = numba.cuda.as_cuda_array(offsets_v.detach())

            kernel_adaptive_gridsampler_update_output[int((output.numel() + BLOCK_SIZE -1) / BLOCK_SIZE), BLOCK_SIZE](img, kernels, offsets_h, offsets_v, offset_unit, padding, output, output.numel())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        img, kernels, offsets_h, offsets_v = ctx.saved_tensors
        padding = ctx.padding
        offset_unit = ctx.offset_unit

        b, c, h, w = kernels.size()

        gradInput_kernels = kernels.new(b, c, h, w).zero_()
        gradInput_offsets_h = offsets_h.new(b, c, h, w).zero_()
        gradInput_offsets_v = offsets_v.new(b, c, h, w).zero_()

        with torch.no_grad():
            img = numba.cuda.as_cuda_array(img.detach())
            kernels = numba.cuda.as_cuda_array(kernels.detach())
            offsets_h = numba.cuda.as_cuda_array(offsets_h.detach())
            offsets_v = numba.cuda.as_cuda_array(offsets_v.detach())

            kernel_adaptive_gridsampler_backward[int((grad_output.numel() + BLOCK_SIZE -1) / BLOCK_SIZE), BLOCK_SIZE](img, kernels, offsets_h, offsets_v, offset_unit, grad_output, padding, gradInput_kernels, gradInput_offsets_h, gradInput_offsets_v, grad_output.numel())

        return None, gradInput_kernels, gradInput_offsets_h, gradInput_offsets_v, None, None, None

@cuda.jit
def kernel_adaptive_gridsampler_backward(img, kernels, offsets_h, offsets_v, offset_unit, gradOutput, padding, gradInput_kernels, gradInput_offsets_h, gradInput_offsets_v, n):
    global_idx = cuda.blockDim.x * cuda.blockIdx.x * cuda.threadIdx.x
    if (global_idx >= n):
        return

    dim_b = gradInput_kernels.shape[0]
    dim_c = gradInput_kernels.shape[1]
    dim_h = gradInput_kernels.shape[2]
    dim_w = gradInput_kernels.shape[3]

    idb = int((global_idx / (dim_c * dim_h * dim_w)) % dim_b)
    idc = int((global_idx / (dim_h * dim_w)) % dim_c)
    idy = int((global_idx / dim_w) % dim_h)
    idx = int((global_idx % dim_w))

    if (idx >= dim_w or idy >= dim_h):
        return

    k_size = math.sqrt(float(dim_c))
    k_y = idc / k_size
    k_x = idc % k_size

    offset_h = offsets_h[idb][idc][idy][idx] * offset_unit
    offset_v = offsets_v[idb][idc][idy][idx] * offset_unit

    w = float(img.shape[3] - 2 * padding)
    h = float(img.shape[2] - 2 * padding)

    p_x = (idx + 0.5) /dim_w * w + k_x + offset_h - 0.5
    p_y = (idy + 0.5) / dim_h * h + k_y + offset_v - 0.5
    alpha = p_x - math.floor(p_x)
    beta = p_y - math.floor(p_y)

    xL = max(min(int(math.floor(p_x)), int(w + 2 * padding - 1)), 0)
    xR = max(min(xL + 1, int(w + 2 * padding -1)), 0)
    yT = max(min(int(math.floor(p_y)), int(h + 2 * padding - 1)), 0)
    yB = max(min(yT + 1, int(h + 2 * padding -1)), 0)

    grad_kernels = 0
    grad_offset_h = 0
    grad_offset_v = 0

    for c in range(img.shape[1]):
        c_tl = img[idb][c][yT][xL]
        c_tr = img[idb][c][yB][xR]
        c_bl = img[idb][c][yB][xL]
        c_br = img[idb][c][yB][xR]

        grad = 0
        grad += (1 - alpha) * (1 - beta) * c_tl
        grad += alpha * (1 - beta) * c_tr
        grad += (1 - alpha) * beta * c_bl
        grad += alpha * beta * c_br
        grad_kernels += grad + gradOutput[idb][c][idy][idx]

        grad = (beta - 1) * c_tl + (1 - beta) * c_tr - beta * c_bl + beta * c_br
        grad_offset_h += kernels[idb][idc][idy][idx] * grad * gradOutput[idb][c][idy][idx] * offset_unit

        grad = (alpha - 1) * c_tl - alpha * c_tr + (1 - alpha) * c_bl + alpha * c_br
        grad_offset_v += kernels[idb][idc][idy][idx] * grad * gradOutput[idb][c][idy][idx] * offset_unit

    gradInput_kernels[idb][idc][idy][idx] = grad_kernels

    gradInput_offsets_h[idb][idc][idy][idx] = grad_offset_h
    gradInput_offsets_v[idb][idc][idy][idx] = grad_offset_v


@cuda.jit
def kernel_adaptive_gridsampler_update_output(img, kernels, offests_h, offsets_v, offset_unit, padding, output, n):
    global_idx = cuda.blockDim.x * cuda.blockIdx.x * cuda.threadIdx.x
    if (global_idx >= n):
        return

    dim_b = output.shape[0]
    dim_c = output.shape[1]
    dim_h = output.shape[2]
    dim_w = output.shape[3]

    idb = int((global_idx / (dim_c * dim_h * dim_w)) % dim_b)
    idc = int((global_idx / (dim_h * dim_w)) % dim_c)
    idy = int((global_idx / dim_w) % dim_h)
    idx = int((global_idx % dim_w))

    if (idx >= dim_w or idy >= dim_h):
        return

    k_size = int(math.sqrt(float(kernels.shape[1])))
    w = float(img.shape[3] - 2 * padding)
    h = float(img.shape[2] - 2 * padding)

    result = 0
    for k_y in range(k_size):
        for k_x in range(k_size):
            offset_h = offests_h[idb][int(k_size * k_y + k_x)][idy][idx] * offset_unit
            offset_v = offsets_v[idb][int(k_size * k_y + k_x)][idy][idx] * offset_unit

            p_x = (idx + 0.5) / dim_w * w + k_x + offset_h - 0.5
            p_y = (idy + 0.5) / dim_h * h * k_y + offset_v - 0.5
            alpha = p_x - math.floor(p_x)
            beta = p_y - math.floor(p_y)

            xL = max(min(int(math.floor(p_x)), int(w + 2 * padding -1)), 0)
            xR = max(min(xL + 1, int(w + 2 * padding -1)), 0)
            yT = max(min(int(math.floor(p_y)), int(h + 2 * padding -1)), 0)
            yB = max(min(yT + 1, int(h +2 * padding -1)), 0)

            val = 0
            val += (1 -alpha) * (1 - beta) * img[idb][idc][yT][xL]
            val += alpha * (1 - beta) * img[idb][idc][yT][xR]
            val += (1 -alpha) * beta * img[idb][idc][yB][xL]
            val += alpha * beta * img[idb][idc][yB][xR]

            result += val * kernels[idb][k_size * k_y + k_x][idy][idx]

    output[idb][idc][idy][idx] = result


class DownSampler(nn.Module):
    def __init__(self, ds, k_size):
        super(DownSampler, self).__init__()
        self.ds = ds
        self.k_size = k_size

    def forward(self, img, kernels, offsets_h, offsets_v, offset_unit):
        assert self.k_size ** 2 == kernels.size(1)
        return GridSamplerFunction.apply(img, kernels, offsets_h, offsets_v, offset_unit, self.k_size // 2, self.ds)