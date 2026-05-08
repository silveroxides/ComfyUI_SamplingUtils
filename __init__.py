import sys
import os
import json
import re
import random
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageFont, ImageSequence
from typing import Union, List, Tuple
import torch
import numpy as np
import scipy
import hashlib
import pilgram
import cv2
import logging
import math
from tqdm import tqdm
from typing_extensions import override
import comfy
from comfy_extras.nodes_logic import SwitchNode, SoftSwitchNode
from comfy_api.latest import ComfyExtension, io
from comfy import model_management
import nodes
import node_helpers
import folder_paths
from nodes import MAX_RESOLUTION
from unifiedefficientloader import UnifiedSafetensorsLoader
from . import system_messages
from . import instruct_prompts
from . import bonus_prompts
from . import edit_target_prompts
from . import edit_op_prompts
from . import camera_shot_prompts

RAM_THRESHOLD = 32 * 1024 * 1024 * 1024 # 16 GB
RAM_FREE_THRESHOLD = 8 * 1024 * 1024 * 1024 # 8 GB

try:
    from unifiedefficientloader import UnifiedSafetensorsLoader
    _unifiedefficientloader_available = True
except ImportError:
    _unifiedefficientloader_available = False

def model_detection_error_hint(path, state_dict):
    filename = os.path.basename(path)
    if 'lora' in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded with a lora loader node.."
    return ""

def load_diffusion_model(unet_path, model_options={}, disable_dynamic=False):
    total_ram = comfy.model_management.get_total_memory(comfy.model_management.torch.device("cpu"))
    free_ram = comfy.model_management.get_free_memory(comfy.model_management.torch.device("cpu"))
    if total_ram < RAM_THRESHOLD or free_ram < RAM_FREE_THRESHOLD or not _unifiedefficientloader_available:
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        logging.warning("Total and free system RAM is low, falling back to ComfyUI default model loading")
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
        if model is None:
            logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
        model.cached_patcher_init = (load_diffusion_model, (unet_path, model_options))
    else:
        with UnifiedSafetensorsLoader(unet_path, low_memory=True) as loader:
            metadata = loader._read_header()
            sd = loader.load_all()
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
        if model is None:
            logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
        model.cached_patcher_init = (load_diffusion_model, (unet_path, model_options))
    return model

def round_to_nearest(n, m):
    return int((n + (m / 2)) // m) * m


# Tensor to PIL
def simpletensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# PIL to Tensor
def simplepil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]


def pad_text_with_joiners(text: str) -> str:
    """
    Pad each character in the text with word joiner unicode characters.

    Parameters:
        text (str): Input string to pad

    Returns:
        str: String with word joiners between each character and at start/end
    """
    if not text:
        return ""

    padding_char = "\u2060"

    # Build the pattern using an f-string to correctly embed the unicode char.
    pattern = f"([^{padding_char}])(?=[^{padding_char}])"
    replacement = r"\1" + padding_char

    joined_text = re.sub(pattern, replacement, text)

    return padding_char + joined_text + padding_char


def ideographic_joined_crlf(text: str) -> str:
    """
    Pad each character in the text with word joiner unicode characters.

    Parameters:
        text (str): Input string to pad

    Returns:
        str: String with word joiners between each character and at start/end
    """
    if not text:
        return ""

    ideographic_pad = "\u2060\u3000\u2060"
    carriage_linefeed = "\u000D\u000A"

    pattern = r"\s"
    replacement = f"{ideographic_pad}{carriage_linefeed}{ideographic_pad}"

    replaced_text = re.sub(pattern, replacement, text)

    return carriage_linefeed + ideographic_pad + replaced_text + ideographic_pad + carriage_linefeed


def ideographic_joined_linepad(text: str) -> str:
    """
    Pad each character in the text with word joiner unicode characters.

    Parameters:
        text (str): Input string to pad

    Returns:
        str: String with word joiners between each character and at start/end
    """
    if not text:
        return ""

    ideographic_pad = "\u2060\u3000\u2060"
    carriage_linefeed = "\u000D\u000A"

    pattern = r"^(.*)$"
    replacement = ideographic_pad + r"\1" + ideographic_pad + carriage_linefeed

    replaced_text = re.sub(pattern, replacement, text)

    return carriage_linefeed + ideographic_pad + replaced_text


def ideographic_joined_sentence(text: str) -> str:
    """
    Pad each character in the text with word joiner unicode characters.

    Parameters:
        text (str): Input string to pad

    Returns:
        str: String with word joiners between each character and at start/end
    """
    if not text:
        return ""

    ideographic_pad = "\u2060\u3000\u2060"
    carriage_linefeed = "\u000D\u000A"

    pattern = r"([^\w\s,]|\.\s)(\w+.+?\.)"
    replacement = r"\1" + carriage_linefeed + ideographic_pad + r"\2" + ideographic_pad + carriage_linefeed

    replaced_text = re.sub(pattern, replacement, text)

    return replaced_text


def to_bold_fraktur(text: str) -> str:
    """
    Convert all ASCII letters in a string to their Unicode mathematical
    bold fraktur counterparts.

    Parameters:
        text (str): Input string to convert

    Returns:
        str: String with letters converted to bold fraktur
    """
    result = []

    # Bold fraktur uppercase starts at U+1D56C (𝕬)
    # Bold fraktur lowercase starts at U+1D586 (𝖆)
    BOLD_FRAKTUR_UPPER_START = 0x1D56C
    BOLD_FRAKTUR_LOWER_START = 0x1D586

    for char in text:
        if "A" <= char <= "Z":
            offset = ord(char) - ord("A")
            result.append(chr(BOLD_FRAKTUR_UPPER_START + offset))
        elif "a" <= char <= "z":
            offset = ord(char) - ord("a")
            result.append(chr(BOLD_FRAKTUR_LOWER_START + offset))
        else:
            result.append(char)

    return "".join(result)


def frakturpad(text: str) -> str:
    """
    Convert ASCII letters to bold fraktur and pad with word joiners.
    First converts A-Z and a-z to their bold fraktur equivalents,
    then pads the result with word joiner characters (U+2060).
    """
    fraktur_text = to_bold_fraktur(text)
    return pad_text_with_joiners(fraktur_text)


def from_bold_fraktur(text: str) -> str:
    """
    Convert all Unicode mathematical bold fraktur letters in a string
    back to their ASCII counterparts.

    Parameters:
        text (str): Input string containing bold fraktur characters

    Returns:
        str: String with bold fraktur letters converted back to ASCII
    """
    result = []

    # Bold fraktur uppercase starts at U+1D56C (𝕬)
    # Bold fraktur lowercase starts at U+1D586 (𝖆)
    BOLD_FRAKTUR_UPPER_START = 0x1D56C
    BOLD_FRAKTUR_UPPER_END = BOLD_FRAKTUR_UPPER_START + 25  # Z
    BOLD_FRAKTUR_LOWER_START = 0x1D586
    BOLD_FRAKTUR_LOWER_END = BOLD_FRAKTUR_LOWER_START + 25  # z

    for char in text:
        code_point = ord(char)
        if BOLD_FRAKTUR_UPPER_START <= code_point <= BOLD_FRAKTUR_UPPER_END:
            offset = code_point - BOLD_FRAKTUR_UPPER_START
            result.append(chr(ord("A") + offset))
        elif BOLD_FRAKTUR_LOWER_START <= code_point <= BOLD_FRAKTUR_LOWER_END:
            offset = code_point - BOLD_FRAKTUR_LOWER_START
            result.append(chr(ord("a") + offset))
        else:
            result.append(char)

    return "".join(result)


def remove_joiners(text: str) -> str:
    """
    Remove all word joiner unicode characters (U+2060) from the text.

    Parameters:
        text (str): Input string potentially containing word joiners

    Returns:
        str: String with all word joiner characters removed
    """
    padding_char = "\u2060"
    return text.replace(padding_char, "")


def unfrakturpad(text: str) -> str:
    """
    Inverse of frakturpad: remove word joiners and convert bold fraktur back to ASCII.
    First removes word joiner characters (U+2060), then converts bold fraktur
    letters back to their A-Z and a-z equivalents.
    """
    text_without_joiners = remove_joiners(text)
    return from_bold_fraktur(text_without_joiners)

def _hex_to_rgb(hex_str: str, default=(255, 255, 255)):
    hex_str = hex_str.lstrip('#')
    try:
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_str) == 8:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4, 6))
    except ValueError:
        pass
    return default

def _diag(H: int, W: int) -> float:
    return math.sqrt(H * H + W * W)

def _pct_to_px(pct: float, diag: float) -> int:
    return max(0, round(abs(pct) * diag / 100.0))

def _blur_kernel_for_diag(diag: float) -> tuple:
    k = max(3, int(round(diag / 724.0 * 3)))
    if k % 2 == 0: k += 1
    return (k, k)

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045,
                   rgb / 12.92,
                   ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        return np.where(t > (6/29)**3,
                        t ** (1/3),
                        t / (3 * (6/29)**2) + 4/29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)], axis=-1).astype(np.float32)

def _dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    return cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)

def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    return (err > threshold).astype(np.float32)

def _grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    if grow_px == 0: return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(np.float32)

def _auto_delta_e_threshold(delta_e: np.ndarray) -> float:
    p75 = float(np.percentile(delta_e, 75))
    p90 = float(np.percentile(delta_e, 90))
    spread = p90 - p75
    threshold = p75 + max(spread * 0.4, 3.0) if spread > 5.0 else p75 + max(spread * 0.6, 4.0)
    return float(np.clip(threshold, 4.0, 60.0))

def _auto_occlusion_threshold(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> float:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    p85 = float(np.percentile(err, 85))
    p95 = float(np.percentile(err, 95))
    threshold = p95 + max((p95 - p85) * 0.5, 0.5)
    return float(np.clip(threshold, 1.0, 15.0))

def _match_histogram(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Adjust the pixel values of a source image such that its histogram
    matches that of a target template image.
    Both source and template should be 2D numpy arrays (a single channel).
    """
    oldshape = source.shape
    source_flat = source.ravel()
    template_flat = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source_flat, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template_flat, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def _match_image_properties(
    original_tensor: torch.Tensor,
    generated_tensor: torch.Tensor,
    overall_weight: float,
    color_weight: float,
    lighting_weight: float,
    texture_preservation: float,
    mask_tensor: torch.Tensor = None,
) -> torch.Tensor:

    batch_size = generated_tensor.size(0)
    out_tensors = []

    orig_batch = original_tensor.size(0)
    mask_batch = mask_tensor.size(0) if mask_tensor is not None else 0

    for i in range(batch_size):
        orig_i = i if i < orig_batch else 0

        orig_np = np.clip(255.0 * original_tensor[orig_i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        gen_np = np.clip(255.0 * generated_tensor[i].cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        mask_np = None
        if mask_tensor is not None:
            mask_i = i if i < mask_batch else 0
            # Extract mask, it might be (H, W) or (1, H, W) or (C, H, W)
            # Typically comfy masks are (H, W)
            m_t = mask_tensor[mask_i].cpu().numpy()
            if m_t.ndim > 2:
                m_t = m_t.squeeze()
            if m_t.shape != gen_np.shape[:2]:
                m_t = cv2.resize(m_t, (gen_np.shape[1], gen_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_np = m_t[:, :, np.newaxis] # (H, W, 1)

        orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2LAB)
        gen_lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB)

        out_lab = np.copy(gen_lab).astype(np.float32)
        gen_lab_f = gen_lab.astype(np.float32)

        # Detail extraction using Bilateral Filter to preserve edges
        # We extract details from the generated image to add them back later
        if texture_preservation > 0.0:
            # Apply bilateral filter to the L channel to get the "base" lighting without textures
            gen_l_base = cv2.bilateralFilter(gen_lab_f[:, :, 0], d=9, sigmaColor=75, sigmaSpace=75)
            # The difference is our high-frequency texture/edge detail
            gen_l_detail = gen_lab_f[:, :, 0] - gen_l_base

            # Apply bilateral filter to the original image L channel as well before matching
            orig_l_base = cv2.bilateralFilter(orig_lab[:, :, 0].astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

            # Match the base (textureless) lighting histograms
            l_trans_base = _match_histogram(gen_l_base, orig_l_base)

            # Add the generated image's original texture back onto the matched base
            l_trans = l_trans_base + (gen_l_detail * texture_preservation)
        else:
            # Standard matching if texture preservation is 0
            l_trans = _match_histogram(gen_lab[:, :, 0], orig_lab[:, :, 0])

        l_weight = lighting_weight * overall_weight
        out_lab[:, :, 0] = gen_lab_f[:, :, 0] * (1.0 - l_weight) + l_trans * l_weight

        # 2. Color (A and B channels)
        # If color_weight > 0, we match the A and B histograms
        a_trans = _match_histogram(gen_lab[:, :, 1], orig_lab[:, :, 1])
        b_trans = _match_histogram(gen_lab[:, :, 2], orig_lab[:, :, 2])
        c_weight = color_weight * overall_weight

        out_lab[:, :, 1] = gen_lab_f[:, :, 1] * (1.0 - c_weight) + a_trans * c_weight
        out_lab[:, :, 2] = gen_lab_f[:, :, 2] * (1.0 - c_weight) + b_trans * c_weight

        # Apply soft masking if provided
        if mask_np is not None:
            out_lab = gen_lab_f * (1.0 - mask_np) + out_lab * mask_np

        out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
        res_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

        out_tensor = torch.from_numpy(res_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        out_tensors.append(out_tensor)

    return torch.cat(out_tensors, dim=0)

def _composite(original_np: np.ndarray,
               generated_np: np.ndarray,
               delta_e_threshold: float,
               flow_preset: int,
               occlusion_threshold: float,
               grow_px: int,
               close_radius: int,
               min_region_px: int,
               feather_px: float) -> tuple:

    H, W = original_np.shape[:2]
    diag = _diag(H, W)

    orig_u8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
    gen_u8  = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    flow_fwd = _dis_flow(gray_orig, gray_gen, flow_preset)
    flow_bwd = _dis_flow(gray_gen, gray_orig, flow_preset)

    warped_gen_dense = _warp(generated_np.astype(np.float32), flow_fwd)

    blur_kernel = _blur_kernel_for_diag(diag)
    orig_blur = cv2.GaussianBlur(original_np, blur_kernel, 0)
    wgen_blur = cv2.GaussianBlur(warped_gen_dense, blur_kernel, 0)

    orig_lab = _rgb_to_lab(orig_blur.reshape(-1, 3)).reshape(H, W, 3)
    wgen_lab = _rgb_to_lab(wgen_blur.reshape(-1, 3)).reshape(H, W, 3)

    lab_diff = orig_lab - wgen_lab
    lab_diff[..., 0] *= 0.7
    delta_e = np.sqrt((lab_diff**2).sum(axis=2))

    sk = max(_blur_kernel_for_diag(diag)[0], 5)
    if sk % 2 == 0: sk += 1
    delta_e_smooth = cv2.GaussianBlur(delta_e, (sk, sk), 0)

    auto_report = {}
    if delta_e_threshold < 0:
        delta_e_threshold = _auto_delta_e_threshold(delta_e_smooth)
        auto_report["auto_delta_e"] = delta_e_threshold

    if occlusion_threshold < 0:
        occlusion_threshold = _auto_occlusion_threshold(flow_fwd, flow_bwd)
        auto_report["auto_occlusion"] = occlusion_threshold

    occluded = _occlusion_mask(flow_fwd, flow_bwd, occlusion_threshold)

    changed = np.maximum((delta_e_smooth > delta_e_threshold).astype(np.float32), occluded)

    if grow_px != 0:
        changed = _grow_mask(changed, grow_px)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if min_region_px > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats((changed > 0.5).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats_cc[i, cv2.CC_STAT_AREA] < min_region_px:
                changed[labeled == i] = 0

    sharp_mask = changed.copy()

    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            fade_dist = feather_px * 3.0
            t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask

    y_grid, x_grid = np.mgrid[0:H:10, 0:W:10]
    pts_orig = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)

    flow_sub = flow_fwd[0:H:10, 0:W:10].reshape(-1, 2)
    mask_sub = sharp_mask[0:H:10, 0:W:10].reshape(-1)

    bg_idx = mask_sub < 0.1
    M = None
    if bg_idx.sum() > 10:
        src_pts = pts_orig[bg_idx]
        dst_pts = src_pts + flow_sub[bg_idx]

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    if M is not None:
        final_aligned_gen = cv2.warpAffine(
            generated_np.astype(np.float32),
            M.astype(np.float64),
            (W, H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
    else:
        final_aligned_gen = generated_np

    m3 = composite_mask[..., np.newaxis]
    result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)

    flow_mag = np.sqrt((flow_fwd**2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int(occluded.sum()),
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e)),
        "resolution":     f"{W}x{H}",
        "diagonal_px":    round(diag),
    }
    stats.update(auto_report)

    return result, composite_mask, stats


class LlamaTokenizerOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlTokenizerOptions",
            category="_for_testing/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("min_padding", default=0, min=0, max=10000, step=1),
                io.Int.Input("min_length", default=0, min=0, max=10000, step=1),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, min_padding, min_length) -> io.NodeOutput:
        clip = clip.clone()
        for llama_type in ["qwen3_4b", "qwen3_8b", "qwen25_7b", "mistral3_24b"]:
            clip.set_tokenizer_option("{}_min_padding".format(llama_type), min_padding)
            clip.set_tokenizer_option("{}_min_length".format(llama_type), min_length)

        return io.NodeOutput(clip)



class SamplingParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplingParameters",
            category="advanced",
            inputs=[
                io.Int.Input(
                    id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(
                    id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
                io.Float.Input(
                    id="scale_by",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="How much to upscale initial resolution by for the upscaled one.",
                ),
                io.Int.Input(
                    id="multiple",
                    default=16,
                    min=4,
                    max=128,
                    step=4,
                    tooltip="Nearest multiple of the result to set the upscaled resolution to.",
                ),
                io.Int.Input(
                    id="steps",
                    default=26,
                    min=1,
                    max=10000,
                    step=1,
                    tooltip="How many steps to run the sampling for.",
                ),
                io.Float.Input(
                    id="cfg",
                    default=3.5,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="The amount of influence your prompot will have on the final image.",
                ),
                io.Int.Input(
                    id="seed",
                    min=-sys.maxsize,
                    max=sys.maxsize,
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="batch_size"),
                io.Int.Output(display_name="upscaled_width"),
                io.Int.Output(display_name="upscaled_height"),
                io.Int.Output(display_name="steps"),
                io.Float.Output(display_name="cfg"),
                io.Int.Output(display_name="seed"),
                io.Int.Output(display_name="tile_width"),
                io.Int.Output(display_name="tile_height"),
                io.Int.Output(display_name="tile_padding"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        batch_size: int = 1,
        scale_by: float,
        multiple: int,
        steps: int,
        cfg: float,
        seed: int,
    ) -> io.NodeOutput:
        upscaled_width = round_to_nearest(int(width * scale_by), int(multiple))
        upscaled_height = round_to_nearest(int(height * scale_by), int(multiple))
        if scale_by > 2.0:
            tile_width = round_to_nearest(
                int((upscaled_width - (width / scale_by)) / scale_by), int(multiple)
            )
            tile_height = round_to_nearest(
                int((upscaled_height - (height / scale_by)) / scale_by), int(multiple)
            )
            tile_padding = round_to_nearest(
                int(max(width, height) - max(tile_width, tile_height)), int(multiple)
            )
        else:
            tile_width = round_to_nearest(int(upscaled_width * 0.5), int(multiple))
            tile_height = round_to_nearest(int(upscaled_height * 0.5), int(multiple))
            tile_padding = round_to_nearest(
                int(max(width, height) - max(tile_width, tile_height)), int(multiple)
            )
        width = round_to_nearest(int(width), int(multiple))
        height = round_to_nearest(int(height), int(multiple))
        return io.NodeOutput(
            width,
            height,
            batch_size,
            upscaled_width,
            upscaled_height,
            steps,
            cfg,
            seed,
            tile_width,
            tile_height,
            tile_padding,
        )

class AdjustedResolutionParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AdjustedResolutionParameters",
            category="utils",
            inputs=[
                io.Int.Input(
                    id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(
                    id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
                io.Float.Input(
                    id="scale_by",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="How much to upscale initial resolution by for the upscaled one.",
                ),
                io.Int.Input(
                    id="multiple",
                    default=16,
                    min=4,
                    max=128,
                    step=4,
                    tooltip="Nearest multiple of the result to set the upscaled resolution to.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="adjusted_width"),
                io.Int.Output(display_name="adjusted_height"),
                io.Int.Output(display_name="upscaled_width"),
                io.Int.Output(display_name="upscaled_height"),
            ],
        )

    @classmethod
    def execute(cls, width: int, height: int, batch_size: int, scale_by: float, multiple: int) -> io.NodeOutput:
        adjusted_width = round_to_nearest(width, multiple)
        adjusted_height = round_to_nearest(height, multiple)
        upscaled_width = round_to_nearest(width * scale_by, multiple)
        upscaled_height = round_to_nearest(height * scale_by, multiple)
        return io.NodeOutput(
            adjusted_width,
            adjusted_height,
            upscaled_width,
            upscaled_height,
        )

class GetJsonKeyValue(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetJsonKeyValue",
            category="advanced/utils",
            inputs=[
                io.String.Input(
                    "json_path",
                    default="./input/JSON_KeyValueStore.json",
                    multiline=False,
                    tooltip="Path to a .json file with simple top level structure with key and value. See example in custom node folder.",
                ),
                io.Combo.Input(
                    "key_id_method",
                    options=["custom", "random_rotate", "increment_rotate"],
                ),
                io.Int.Input(
                    "rotation_interval",
                    default=0,
                    tooltip="how many steps to jump when doing rotate.",
                ),
                io.String.Input(
                    "key_id",
                    default="placeholder",
                    multiline=False,
                    tooltip="Put name of key in the .json here if using custom in key_id_method.",
                ),
            ],
            outputs=[io.String.Output(display_name="key_value")],
        )

    @classmethod
    def execute(
        cls, json_path, key_id_method, rotation_interval, key_id="placeholder"
    ) -> io.NodeOutput:
        """
        Loads API keys from a JSON file (top-level dictionary)
        and selects one based on the specified method.

        Args:
            json_path (str): Path to the JSON file. Expected format:
                             {"key_id_1": "api_key_value_1", "key_id_2": "api_key_value_2", ...}
            key_id_method (str): Method to select the key ('custom', 'random_rotate', 'increment_rotate').
            rotation_interval (int): Used as index for 'increment_rotate'.
            key_id (str, optional): ID (key name) of the key to select if key_id_method is 'custom'. Defaults to "placeholder".

        Returns:
            str: The selected API key string.
            Raises: ValueError or RuntimeError if unable to find or select a key.
        """
        api_keys_data = None
        absolute_json_path = os.path.abspath(json_path)

        try:
            with open(absolute_json_path, "r") as f:
                api_keys_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(
                f"RotateKeyAPI Error: JSON file not found at {absolute_json_path}"
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"RotateKeyAPI Error: Could not decode JSON from {absolute_json_path}. Check file format."
            )
        except Exception as e:
            raise RuntimeError(
                f"RotateKeyAPI Error: Unexpected error reading file {absolute_json_path}: {e}"
            )

        if not isinstance(api_keys_data, dict):
            raise ValueError(
                f"RotateKeyAPI Error: JSON content is not a dictionary in {absolute_json_path}. Expected format: {{'key_id': 'api_key', ...}}"
            )

        if not api_keys_data:
            raise ValueError(
                f"RotateKeyAPI Error: The JSON dictionary in {absolute_json_path} is empty."
            )

        selected_key_value = None

        if key_id_method == "custom":
            if key_id == "placeholder":
                print(
                    "RotateKeyAPI Warning: 'custom' method selected but 'key_id' is still the default 'placeholder'. Ensure this is intended or provide a valid key ID."
                )

            selected_key_value = api_keys_data.get(key_id)

            if selected_key_value is None:
                raise ValueError(
                    f"RotateKeyAPI Error: Custom key ID '{key_id}' not found in the JSON dictionary keys."
                )

        elif key_id_method == "random_rotate":
            api_keys_list = list(api_keys_data.values())

            selected_key_value = random.choice(api_keys_list)

        elif key_id_method == "increment_rotate":
            api_keys_list = list(api_keys_data.values())

            index = rotation_interval % len(api_keys_list)

            try:
                selected_key_value = api_keys_list[index]
            except IndexError:
                raise IndexError(
                    f"RotateKeyAPI Error: Calculated index {index} (from interval {rotation_interval}) is out of bounds for list of size {len(api_keys_list)}."
                )
            except Exception as e:
                raise RuntimeError(
                    f"RotateKeyAPI Error: Unexpected error accessing item at index {index}: {e}"
                )

        if not isinstance(selected_key_value, str) or not selected_key_value:
            raise ValueError(
                f"RotateKeyAPI Error: Retrieved value for selected key is not a valid string. Value: {selected_key_value}"
            )

        print(
            f"RotateKeyAPI: Successfully retrieved API key using method '{key_id_method}'."
        )
        return io.NodeOutput(selected_key_value)


class Image_Color_Noise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image_Color_Noise",
            category="utils",
            inputs=[
                io.Int.Input("width", default=512, max=4096, min=64, step=1),
                io.Int.Input("height", default=512, max=4096, min=64, step=1),
                io.Float.Input("frequency", default=0.5, max=100.0, min=0.0, step=0.01),
                io.Float.Input(
                    "attenuation", default=0.5, max=100.0, min=0.0, step=0.01
                ),
                io.Combo.Input(
                    "noise_type",
                    options=["grey", "white", "red", "pink", "green", "blue", "mix"],
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[
                io.Image.Output(display_name="noise_image"),
            ],
        )

    @classmethod
    def execute(cls, width, height, frequency, attenuation, noise_type, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise_image = cls.generate_power_noise(
            width, height, frequency, attenuation, noise_type, generator
        )
        return io.NodeOutput(pil2tensor(noise_image))

    @classmethod
    def generate_power_noise(
        cls, width, height, frequency, attenuation, noise_type, generator
    ):
        def normalize_array(arr):
            return (255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))).astype(
                np.uint8
            )

        def white_noise(w, h, gen):
            return torch.rand(h, w, generator=gen).numpy()

        def grey_noise_texture(w, h, att, gen):
            return torch.normal(mean=0, std=att, size=(h, w), generator=gen).numpy()

        def fourier_noise(w, h, att, power_modifier, gen):
            noise = grey_noise_texture(w, h, att, gen)
            fy = np.fft.fftfreq(h)[:, np.newaxis]
            fx = np.fft.fftfreq(w)
            f = np.sqrt(fx**2 + fy**2)
            f[0, 0] = 1.0
            power_spectrum = f**power_modifier
            fft_noise = np.fft.fft2(noise)
            fft_modified = fft_noise * power_spectrum
            inv_fft = np.fft.ifft2(fft_modified)
            return np.real(inv_fft)

        noise_array = np.zeros((height, width, 3), dtype=np.uint8)
        zeros_channel = np.zeros((height, width), dtype=np.uint8)

        if noise_type == "grey":
            luma = normalize_array(
                grey_noise_texture(width, height, attenuation, generator)
            )
            noise_array = np.stack([luma, luma, luma], axis=-1)

        elif noise_type == "white":
            r = normalize_array(white_noise(width, height, generator))
            g = normalize_array(white_noise(width, height, generator))
            b = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([r, g, b], axis=-1)

        elif noise_type == "red":
            r = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([r, zeros_channel, zeros_channel], axis=-1)

        elif noise_type == "green":
            g = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([zeros_channel, g, zeros_channel], axis=-1)

        elif noise_type == "blue":
            b = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([zeros_channel, zeros_channel, b], axis=-1)

        elif noise_type == "pink":
            base_texture = fourier_noise(width, height, attenuation, -1.0, generator)
            r = normalize_array(base_texture)
            g = (r * 0.75).astype(np.uint8)
            b = (r * 0.85).astype(np.uint8)
            noise_array = np.stack([r, g, b], axis=-1)

        elif noise_type == "mix":
            r = normalize_array(
                fourier_noise(width, height, attenuation, -1.0, generator)
            )  # Pink Frequency
            g = normalize_array(
                fourier_noise(width, height, attenuation, 0.5, generator)
            )  # Green Frequency
            b = normalize_array(
                fourier_noise(width, height, attenuation, 1.0, generator)
            )  # Blue Frequency
            noise_array = np.stack([r, g, b], axis=-1)

        else:
            print(f"[ERROR] Unsupported noise type `{noise_type}`")
            return Image.new("RGB", (width, height), color="black")

        return Image.fromarray(noise_array, "RGB")


class SystemMessagePresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(system_messages):
            if name.startswith("SYSTEM_MESSAGE"):
                val = getattr(system_messages, name)
                if isinstance(val, str):
                    if name == "SYSTEM_MESSAGE":
                        presets["F2_SYSTEM_MESSAGE"] = val
                    elif name == "SYSTEM_MESSAGE_UPSAMPLING_I2I":
                        presets["F2_SYSTEM_MESSAGE_UPSAMPLING_I2I"] = val
                    elif name == "SYSTEM_MESSAGE_UPSAMPLING_T2I":
                        presets["F2_SYSTEM_MESSAGE_UPSAMPLING_T2I"] = val
                    elif name.startswith("SYSTEM_MESSAGE_STYLE_"):
                        presets[name.replace("SYSTEM_MESSAGE_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="SystemMessagePresets",
            category="advanced/text",
            display_name="System Message Presets",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=sorted(list(presets.keys())),
                    default="F2_SYSTEM_MESSAGE" if "F2_SYSTEM_MESSAGE" in presets else sorted(list(presets.keys()))[0],
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        system_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(system_prompt)


class InstructPromptPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(instruct_prompts):
            if name.startswith("INSTRUCT_PROMPT_STYLE_"):
                val = getattr(instruct_prompts, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="InstructPromptPresets",
            category="advanced/text",
            display_name="Instruct Prompt Presets",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=sorted(list(presets.keys())),
                    default=sorted(list(presets.keys()))[0] if presets else "",
                ),
            ],
            outputs=[
                io.String.Output(display_name="instruct_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        instruct_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(instruct_prompt)


class BonusPromptPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(bonus_prompts):
            if name.startswith("BONUS_PROMPT_STYLE_"):
                val = getattr(bonus_prompts, name)
                if isinstance(val, str):
                    presets[name.replace("BONUS_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="BonusPromptPresets",
            category="advanced/text",
            display_name="Bonus Prompt Presets",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=sorted(list(presets.keys())),
                    default=sorted(list(presets.keys()))[0] if presets else "",
                ),
            ],
            outputs=[
                io.String.Output(display_name="bonus_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        bonus_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(bonus_prompt)


class EditTargetPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(edit_target_prompts):
            if name.startswith("BONUS_PROMPT_EDIT_TARGET_"):
                val = getattr(edit_target_prompts, name)
                if isinstance(val, str):
                    presets[name.replace("BONUS_PROMPT_EDIT_TARGET_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="EditTargetPresets",
            category="advanced/text",
            display_name="Edit Target Presets",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=sorted(list(presets.keys())),
                    default=sorted(list(presets.keys()))[0] if presets else "",
                ),
            ],
            outputs=[
                io.String.Output(display_name="edit_target_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        edit_target_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(edit_target_prompt)


class EditOpPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(edit_op_prompts):
            if name.startswith("INSTRUCT_PROMPT_EDIT_OP_"):
                val = getattr(edit_op_prompts, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_EDIT_OP_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="EditOpPresets",
            category="advanced/text",
            display_name="Edit Operation Presets",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=sorted(list(presets.keys())),
                    default=sorted(list(presets.keys()))[0] if presets else "",
                ),
            ],
            outputs=[
                io.String.Output(display_name="edit_op_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        edit_op_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(edit_op_prompt)


class CameraShotPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(camera_shot_prompts):
            if name.startswith("INSTRUCT_PROMPT_CAMERA_SHOT_"):
                val = getattr(camera_shot_prompts, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_CAMERA_SHOT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="CameraShotPresets",
            category="advanced/text",
            display_name="Camera Shot Presets",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=sorted(list(presets.keys())),
                    default=sorted(list(presets.keys()))[0] if presets else "",
                ),
            ],
            outputs=[
                io.String.Output(display_name="camera_shot_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        camera_shot_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(camera_shot_prompt)


class VLMSysInstrPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        json_path = os.path.join(os.path.dirname(__file__), "system_instructions_vlm.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
        except Exception as e:
            print(f"Error loading VLMSysInstrPresets: {e}")
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(list(presets.keys()))
        default = options[0] if options else ""
        return io.Schema(
            node_id="VLMSysInstrPresets",
            display_name="VLM System Instruction Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=options,
                    default=default,
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_instruction"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        system_instruction = presets_dict.get(preset, "")
        return io.NodeOutput(system_instruction)


class VLMSysQueryAddPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        base_names = set()
        json_path = os.path.join(os.path.dirname(__file__), "system_query_additional_vlm.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key in data:
                    base_names.add(key.removesuffix("_prefix").removesuffix("_suffix"))
        except Exception as e:
            print(f"Error loading VLMSysQueryAddPresets: {e}")
        return sorted(list(base_names))

    @classmethod
    def define_schema(cls):
        options = cls.get_presets()
        default = options[0] if options else ""
        return io.Schema(
            node_id="VLMSysQueryAddPresets",
            display_name="VLM System Query Add Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=options,
                    default=default,
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_query_additional"),
            ],
        )

    @classmethod
    def execute(cls, preset, text) -> io.NodeOutput:
        json_path = os.path.join(os.path.dirname(__file__), "system_query_additional_vlm.json")
        prefix = ""
        suffix = ""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                prefix = data.get(f"{preset}_prefix", "")
                suffix = data.get(f"{preset}_suffix", "")
        except Exception as e:
            print(f"Error executing VLMSysQueryAddPresets: {e}")

        return io.NodeOutput(f"{prefix}{text}{suffix}")


class VLMSysInstrAdvPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        json_path = os.path.join(
            os.path.dirname(__file__), "system_instructions_vlm.json"
        )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                presets = json.load(f)
        except Exception as e:
            print(f"Error loading VLMSysInstrPresets: {e}")
        return presets

    @classmethod
    def get_additional_instructions(cls):
        instructions = {}
        json_path = os.path.join(
            os.path.dirname(__file__), "additional_instructions_vlm.json"
        )
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                instructions = json.load(f)
        except Exception as e:
            print(f"Error loading additional_instructions_vlm: {e}")
        return instructions

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(list(presets.keys()))
        default = options[0] if options else ""
        return io.Schema(
            node_id="VLMSysInstrAdvPresets",
            display_name="VLM System Instruction Advanced Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=options,
                    default=default,
                ),
                io.Boolean.Input("jailbreak", default=False),
                io.String.Input("system_query", multiline=True, default=""),
                io.String.Input("user_query", multiline=True, default=""),
            ],
            outputs=[
                io.String.Output(display_name="system_instruction"),
            ],
        )

    @classmethod
    def execute(cls, preset, jailbreak, system_query, user_query) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        additional = cls.get_additional_instructions()

        system_instruction = presets_dict.get(preset, "")

        # Start building the final output
        # Formula: {jailbreak_prefix} + {chosen_preset_content} + {system_query_prefix} + {system_query_string} + {user_query_prefix} + {user_query_string} + {user_query_suffix} + {system_query_suffix} + {jailbreak_suffix}

        result = system_instruction

        # Apply system query wrapping
        if system_query:
            sq_prefix = additional.get("system_query_prefix", "")
            sq_suffix = additional.get("system_query_suffix", "")
            result = result + sq_prefix + system_query

            # Apply user query nesting
            if user_query:
                uq_prefix = additional.get("user_query_prefix", "")
                uq_suffix = additional.get("user_query_suffix", "")
                result = result + uq_prefix + user_query + uq_suffix

            result = result + sq_suffix
        elif user_query:
            # If system_query is empty but user_query is not, just add user query components
            uq_prefix = additional.get("user_query_prefix", "")
            uq_suffix = additional.get("user_query_suffix", "")
            result = result + uq_prefix + user_query + uq_suffix

        # Apply jailbreak wrapping
        if jailbreak:
            jb_prefix = additional.get("jailbreak_prefix", "")
            jb_suffix = additional.get("jailbreak_suffix", "")
            result = jb_prefix + result + jb_suffix

        return io.NodeOutput(result)


class UnifiedPresets(io.ComfyNode):
    """
    Primitive node that unifies shared presets between SystemMessagePresets,
    InstructPromptPresets, and BonusPromptPresets.
    Outputs selected preset as 'any' type for flexible downstream usage.
    """

    @classmethod
    def get_shared_presets(cls):
        """Get presets that are shared between all three preset sources"""
        system_presets = set(SystemMessagePresets.get_presets().keys())
        instruct_presets = set(InstructPromptPresets.get_presets().keys())
        bonus_presets = set(BonusPromptPresets.get_presets().keys())

        # Find intersection of all three
        shared = system_presets & instruct_presets & bonus_presets
        return sorted(list(shared))

    @classmethod
    def define_schema(cls) -> io.Schema:
        shared_presets = cls.get_shared_presets()
        default_preset = shared_presets[0] if shared_presets else ""

        return io.Schema(
            node_id="UnifiedPresets",
            display_name="Unified Presets (Primitive)",
            category="advanced/primitives",
            inputs=[
                io.Combo.Input(
                    "preset",
                    options=shared_presets,
                    default=default_preset,
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="preset"),
            ],
        )

    @classmethod
    def execute(cls, preset: str) -> io.NodeOutput:
        """Forward the selected preset as 'any' type"""
        return io.NodeOutput(preset)


class TextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", image=None) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n"
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n"
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        if image is not None:
            tokens = clip.tokenize(prompt, llama_template=llama_template, image=image)
        else:
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


# Template definitions for unified node
SYSTEM_PROMPT_TEMPLATES = {
    "flux2dev": {
        "prefix": r"[SYSTEM_PROMPT]",
        "suffix": r"[/SYSTEM_PROMPT][INST]{}[/INST]",
    },
    "klein": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "z-image": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    },
}


class TextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeSystemPrompt",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "z-image"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="(Klein only) Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        if model_type == "klein" and len(thinking_content) > 0:
            # Klein with custom thinking content
            if len(system_prompt) > 0:
                llama_template = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{{}}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            else:
                llama_template = (
                    "<|im_start|>user\n{}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class ModifyMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModifyMask",
            category="utils/mask",
            display_name="Modify Mask (Expand/Contract with options)",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input(
                    "expand", default=0, max=MAX_RESOLUTION, min=-MAX_RESOLUTION, step=1
                ),
                io.Float.Input(
                    "incremental_expandrate", default=0.0, max=100.0, min=0.0, step=0.01
                ),
                io.Boolean.Input("tapered_corners", default=True),
                io.Boolean.Input("flip_input", default=False),
                io.Float.Input(
                    "blur_radius", default=0.0, max=100.0, min=0.0, step=0.01
                ),
                io.Float.Input("lerp_alpha", default=1.0, max=1.0, min=0.0, step=0.01),
                io.Float.Input(
                    "decay_factor", default=1.0, max=1.0, min=0.0, step=0.01
                ),
                io.Boolean.Input("fill_holes", default=False, optional=True),
                io.Float.Input("lower_clamp", default=0.0, max=100.0, min=0.0, step=0.1),
                io.Float.Input("upper_clamp", default=100.0, max=100.0, min=0.0, step=0.1),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Mask.Output(display_name="mask_inverted"),
            ],
        )

    @classmethod
    def execute(
        self,
        mask,
        expand,
        tapered_corners,
        flip_input,
        blur_radius,
        incremental_expandrate,
        lerp_alpha,
        decay_factor,
        fill_holes=False,
        lower_clamp=0.0,
        upper_clamp=100.0,
    ):
        import kornia.morphology as morph

        alpha = lerp_alpha
        decay = decay_factor

        # 1. Clone the original mask to keep a reference to the un-blurred pixels
        original_mask_input = mask.clone()

        if flip_input:
            mask = 1.0 - mask
            original_mask_input = 1.0 - original_mask_input

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))

        # Prepare original mask for processing loop (match dimensions)
        original_mask_batches = original_mask_input.reshape(
            (-1, mask.shape[-2], mask.shape[-1])
        )

        out = []
        previous_output = None
        current_expand = expand
        for m in tqdm(growmask, desc="Expanding/Contracting Mask"):
            output = (
                m.unsqueeze(0).unsqueeze(0).to(model_management.get_torch_device())
            )  # Add batch and channel dims for kornia
            if abs(round(current_expand)) > 0:
                # Create kernel - kornia expects kernel on same device as input
                if tapered_corners:
                    kernel = torch.tensor(
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        dtype=torch.float32,
                        device=output.device,
                    )
                else:
                    kernel = torch.tensor(
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=torch.float32,
                        device=output.device,
                    )

                for _ in range(abs(round(current_expand))):
                    if current_expand < 0:
                        output = morph.erosion(output, kernel)
                    else:
                        output = morph.dilation(output, kernel)

            output = output.squeeze(0).squeeze(0)  # Remove batch and channel dims

            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)

            if fill_holes:
                binary_mask = output > 0
                output_np = binary_mask.cpu().numpy()
                filled = scipy.ndimage.binary_fill_holes(output_np)
                output = torch.from_numpy(filled.astype(np.float32)).to(output.device)

            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output.cpu())

        if blur_radius != 0 and current_expand != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                blurred_tensor = pil2tensor(pil_image)

                # 2. Restore the original pixels IF we are expanding
                # We use torch.max: this keeps the original pixel value unless the
                # blurred expansion is brighter. It prevents "adding" values together.
                if current_expand > 0:
                    original_slice = original_mask_batches[idx].unsqueeze(0).cpu()
                    blurred_tensor = torch.max(blurred_tensor, original_slice)
                else:
                    original_slice = original_mask_batches[idx].unsqueeze(0).cpu()
                    blurred_tensor = torch.min(blurred_tensor, original_slice)

                out[idx] = blurred_tensor

            blurred = torch.cat(out, dim=0)
            if lower_clamp > 0.0:
                blurred = torch.max(blurred, torch.tensor(lower_clamp / 100.0, device=blurred.device))
            if upper_clamp < 100.0:
                blurred = torch.min(blurred, torch.tensor(upper_clamp / 100.0, device=blurred.device))
            mask = blurred
            mask_inverted = 1.0 - blurred
            return io.NodeOutput(mask, mask_inverted)
        elif blur_radius != 0 and current_expand == 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            if lower_clamp > 0.0:
                blurred = torch.max(blurred, torch.tensor(lower_clamp / 100.0, device=blurred.device))
            if upper_clamp < 100.0:
                blurred = torch.min(blurred, torch.tensor(upper_clamp / 100.0, device=blurred.device))
            mask = blurred
            mask_inverted = 1.0 - blurred
            return io.NodeOutput(mask, mask_inverted)
        else:
            mask = torch.stack(out, dim=0)
            if lower_clamp > 0.0:
                mask = torch.max(mask, torch.tensor(lower_clamp / 100.0, device=mask.device))
            if upper_clamp < 100.0:
                mask = torch.min(mask, torch.tensor(upper_clamp / 100.0, device=mask.device))
            mask_inverted = 1.0 - mask
            return io.NodeOutput(mask, mask_inverted)


class ImageBlendByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageBlendByMask",
            category="utils/mask",
            display_name="Image Blend by Mask",
            inputs=[
                io.Image.Input("destination"),
                io.Image.Input("source"),
                io.Combo.Input(
                    "mode",
                    options=[
                        "add",
                        "color",
                        "color_burn",
                        "color_dodge",
                        "darken",
                        "difference",
                        "exclusion",
                        "hard_light",
                        "hue",
                        "lighten",
                        "multiply",
                        "overlay",
                        "screen",
                        "soft_light",
                    ],
                    default="add",
                ),
                io.Float.Input(
                    "blend_percentage", default=1.0, max=1.0, min=0.0, step=0.01
                ),
                io.Boolean.Input("resize_source", default=False),
                io.Mask.Input("mask"),
            ],
            outputs=[
                io.Image.Output(display_name="blended_image"),
            ],
        )

    @classmethod
    def execute(
        self,
        destination,
        source,
        mode="add",
        blend_percentage=1.0,
        resize_source=False,
        mask=None,
    ):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        source = source.movedim(-1, 1).to(destination.device)

        if resize_source:
            source = torch.nn.functional.interpolate(
                source,
                size=(destination.shape[-2], destination.shape[-1]),
                mode="bicubic",
            )

        # Convert images to PIL
        img_a = simpletensor2pil(destination)
        img_b = simpletensor2pil(source)

        # Apply blending mode
        blending_modes = {
            "color": pilgram.css.blending.color,
            "color_burn": pilgram.css.blending.color_burn,
            "color_dodge": pilgram.css.blending.color_dodge,
            "darken": pilgram.css.blending.darken,
            "difference": pilgram.css.blending.difference,
            "exclusion": pilgram.css.blending.exclusion,
            "hard_light": pilgram.css.blending.hard_light,
            "hue": pilgram.css.blending.hue,
            "lighten": pilgram.css.blending.lighten,
            "multiply": pilgram.css.blending.multiply,
            "add": pilgram.css.blending.normal,
            "overlay": pilgram.css.blending.overlay,
            "screen": pilgram.css.blending.screen,
            "soft_light": pilgram.css.blending.soft_light,
        }

        out_image = blending_modes.get(mode, pilgram.css.blending.normal)(img_a, img_b)

        out_image = out_image.convert("RGB")

        # Apply mask if provided
        if mask is not None:
            mask = ImageOps.invert(simpletensor2pil(mask).convert("L"))
            out_image = Image.composite(img_a, out_image, mask.resize(img_a.size))

        # Blend image based on blend percentage
        blend_mask = Image.new(
            mode="L", size=img_a.size, color=(round(blend_percentage * 255))
        )
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(img_a, out_image, blend_mask)

        blended_image = simplepil2tensor(out_image)
        return io.NodeOutput(blended_image)


class SU_InjectNoiseToLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SU_InjectNoiseToLatent",
            category="utils/latent",
            display_name="Inject Noise to Latent SamplingUtils",
            inputs=[
                io.Latent.Input("latents"),
                io.Float.Input("strength", default=0.1, min=0.0, max=200.0, step=0.0001),
                io.Latent.Input("noise"),
                io.Boolean.Input("normalize", default=False),
                io.Boolean.Input("average", default=False),
                io.Mask.Input("mask", optional=True),
                io.Float.Input("mix_randn_amount", default=0.0, min=0.0, max=1000.0, step=0.001, optional=True),
                io.Int.Input("seed", default=123, min=0, max=0xffffffffffffffff, step=1, optional=True),
            ],
            outputs=[
                io.Latent.Output("noised_latents"),
            ],
        )

    @classmethod
    def execute(
            cls, latents, strength,
            noise, normalize, average,
            mix_randn_amount=0, seed=None, mask=None
        ) -> io.NodeOutput:
        samples = latents["samples"].clone().cpu()
        noise = noise["samples"].clone().cpu()

        # Handle potential 4D vs 5D mismatch (Image vs Video latents)
        if len(samples.shape) != len(noise.shape):
            if len(samples.shape) == 5 and len(noise.shape) == 4:
                # Samples is [B, C, T, H, W], noise is [B, C, H, W]
                # Unsqueeze noise to [B, C, 1, H, W] to broadcast over T
                noise = noise.unsqueeze(2)
            elif len(samples.shape) == 4 and len(noise.shape) == 5:
                # Samples is [B, C, H, W], noise is [B, C, T, H, W]
                if noise.shape[2] == 1:
                    noise = noise.squeeze(2)
                else:
                    # Take first frame of noise
                    noise = noise[:, :, 0, :, :]

        # Match Channels
        if samples.shape[1] != noise.shape[1]:
            if noise.shape[1] == 1:
                noise = noise.expand(-1, samples.shape[1], *([-1] * (len(noise.shape) - 2)))
            else:
                # Repeat or truncate channels to match
                if noise.shape[1] < samples.shape[1]:
                    noise = noise.repeat(1, (samples.shape[1] // noise.shape[1]) + 1, *([1] * (len(noise.shape) - 2)))
                noise = noise[:, :samples.shape[1], ...]

        # Match Batch Size
        if samples.shape[0] != noise.shape[0]:
            if noise.shape[0] == 1:
                noise = noise.expand(samples.shape[0], *([-1] * (len(noise.shape) - 1)))
            else:
                if noise.shape[0] < samples.shape[0]:
                    noise = noise.repeat((samples.shape[0] // noise.shape[0]) + 1, *([1] * (len(noise.shape) - 1)))
                noise = noise[:samples.shape[0], ...]

        # Match Width/Height (Spatial dimensions)
        if samples.shape[-2:] != noise.shape[-2:]:
            # Spatial mismatch - resize noise to match samples
            if len(noise.shape) == 5:
                # For 5D, we interpolate frame by frame
                B, C, T, H, W = noise.shape
                noise = noise.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                noise = torch.nn.functional.interpolate(noise, size=samples.shape[-2:], mode="bicubic")
                noise = noise.reshape(B, T, C, *samples.shape[-2:]).permute(0, 2, 1, 3, 4)
            else:
                noise = torch.nn.functional.interpolate(noise, size=samples.shape[-2:], mode="bicubic")

        if average:
            noised = (samples + noise) / 2
        else:
            noised = samples + noise * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            # Match mask to spatial dimensions [H, W]
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noised.shape[-2], noised.shape[-1]), mode="bilinear")

            if len(noised.shape) == 5:
                # mask is [B, 1, H, W], expand to [B, C, T, H, W]
                mask = mask.unsqueeze(2).expand((-1, noised.shape[1], noised.shape[2], -1, -1))
            else:
                # mask is [B, 1, H, W], expand to [B, C, H, W]
                mask = mask.expand((-1, noised.shape[1], -1, -1))

            if mask.shape[0] < noised.shape[0]:
                mask = mask.repeat((noised.shape[0] - 1) // mask.shape[0] + 1, *([1] * (len(mask.shape) - 1)))[:noised.shape[0]]
            noised = mask * noised + (1-mask) * samples
        if mix_randn_amount > 0:
            if seed is not None:
                generator = torch.manual_seed(seed)
                rand_noise = torch.randn(noised.size(), dtype=noised.dtype, layout=noised.layout, generator=generator, device="cpu")
                noised = noised + (mix_randn_amount * rand_noise)

        return io.NodeOutput({"samples": noised.to(latents["samples"].device)})


class FrakturPadNode(io.ComfyNode):
    """
    ComfyUI node that obfuscate text to some systems by converting text to bold fraktur with word joiner padding.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Frakturpad",
            display_name="Frakturpad (Text Obfuscation)",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to convert...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="fraktur_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        """
        Execute the frakturpad conversion.
        """
        result = frakturpad(text)
        return io.NodeOutput(result)


class UnFrakturPadNode(io.ComfyNode):
    """
    ComfyUI node that deobfuscates text by converting bold fraktur back to ASCII
    and removing word joiner padding. This is the inverse of FrakturPadNode.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UnFrakturPad",
            display_name="UnFrakturPad (Text Deobfuscation)",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter obfuscated text to convert back...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="plain_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        """
        Execute the unfrakturpad conversion.
        """
        result = unfrakturpad(text)
        return io.NodeOutput(result)


class JoinerPadding(io.ComfyNode):
    """
    ComfyUI node that pads text with word joiner characters.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JoinerPadding",
            display_name="Joiner Padding",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to pad...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="padded_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        """
        Execute the joiner padding.
        """
        result = pad_text_with_joiners(text)
        return io.NodeOutput(result)


class IdeographicTagPad(io.ComfyNode):
    """
    ComfyUI node that obfuscate text to some systems by converting text to bold fraktur with word joiner padding.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IdeographicTagPad",
            display_name="IdeographicTagPad (Text Obfuscation)",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to convert...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="padded_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        """
        Execute the frakturpad conversion.
        """
        result = ideographic_joined_crlf(text)
        return io.NodeOutput(result)


class IdeographicLinePad(io.ComfyNode):
    """
    ComfyUI node that obfuscate text to some systems by converting text to bold fraktur with word joiner padding.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IdeographicLinePad",
            display_name="IdeographicLinePad (Text Obfuscation)",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to convert...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="padded_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        """
        Execute the frakturpad conversion.
        """
        result = ideographic_joined_linepad(text)
        return io.NodeOutput(result)


class IdeographicSentencePad(io.ComfyNode):
    """
    ComfyUI node that obfuscate text to some systems by converting text to bold fraktur with word joiner padding.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IdeographicSentencePad",
            display_name="IdeographicSentencePad (Text Obfuscation)",
            category="advanced/text",
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder="Enter text to convert...",
                ),
            ],
            outputs=[
                io.String.Output(display_name="padded_text"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        """
        Execute the frakturpad conversion.
        """
        result = ideographic_joined_sentence(text)
        return io.NodeOutput(result)


class SU_LoadImagePath(io.ComfyNode):
    """
    Load an image from an arbitrary file path with proper mask handling.
    Returns the image and a mask extracted from the alpha channel.
    For images without alpha, returns a full-sized zero mask (not 64x64).
    Supports both absolute and relative paths, with any OS path separator.
    """

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Normalize a file path to handle:
        - Backslashes (Windows) and forward slashes (Unix)
        - Relative paths (starting with '.', '..', or lowercase letter)
        - Whitespaces in paths and filenames

        Returns an absolute, normalized path.
        """
        if not path:
            return path

        # Strip leading/trailing whitespace but preserve internal whitespace
        path = path.strip()

        # Normalize path separators: replace backslashes with forward slashes
        # Then use os.path.normpath to get the OS-appropriate format
        path = path.replace('\\', '/')
        path = os.path.normpath(path)

        # Convert to absolute path if relative
        # os.path.abspath handles: '.', '..', and paths without drive letter
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        return path

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SU_LoadImagePath",
            display_name="Load Image (Path)",
            category="advanced/image",
            inputs=[
                io.String.Input(
                    "image_path",
                    multiline=False,
                    placeholder="path/to/image.png or X:/path/to/image.png",
                    tooltip="Path to the image file. Supports absolute or relative paths with any OS format (backslashes or forward slashes). Whitespaces in paths are supported.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Mask.Output(display_name="MASK"),
                io.Mask.Output(display_name="MASK_INVERTED"),
            ],
        )

    @classmethod
    def execute(cls, image_path: str) -> io.NodeOutput:
        # Normalize the path to handle relative paths and different separators
        normalized_path = cls._normalize_path(image_path)

        # Validate path
        if not normalized_path or not os.path.isfile(normalized_path):
            raise ValueError(f"Invalid image path: {image_path} (resolved to: {normalized_path})")

        img = node_helpers.pillow(Image.open, normalized_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            # Handle 16-bit images (mode 'I') - normalize by 65535, not 255
            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 65535))

            image = i.convert("RGB")

            # Set dimensions from first frame
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            # Skip frames with different dimensions
            if image.size[0] != w or image.size[1] != h:
                continue

            # Convert to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            # Extract mask from alpha channel
            if 'A' in i.getbands():
                # RGBA image - extract alpha
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            elif i.mode == 'P' and 'transparency' in i.info:
                # Palette mode with transparency - convert to RGBA (already transposed)
                rgba = i.convert('RGBA')
                mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            else:
                # No alpha - return full-sized zero mask (NOT 64x64!)
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

            # MPO format: only use first frame
            if img.format == "MPO":
                break

        if len(output_images) == 0:
            raise ValueError(f"No valid image frames could be loaded from: {image_path}")

        # Stack frames
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Create inverted mask
        output_mask_inverted = 1.0 - output_mask

        return io.NodeOutput(output_image, output_mask, output_mask_inverted)

    @classmethod
    def IS_CHANGED(cls, image_path: str):
        normalized_path = cls._normalize_path(image_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            return ""
        m = hashlib.sha256()
        with open(normalized_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image_path: str):
        if not image_path:
            return "Image path cannot be empty"
        normalized_path = cls._normalize_path(image_path)
        if not os.path.isfile(normalized_path):
            return f"Invalid image file: {image_path} (resolved to: {normalized_path})"
        return True


class SU_LoadImageDirectory(io.ComfyNode):
    """
    Load multiple images from a directory as a batch.
    Supports selecting a range of images using start index and count.
    Images are sorted alphanumerically.
    """

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return path
        path = path.strip()
        path = path.replace('\\', '/')
        path = os.path.normpath(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SU_LoadImageDirectory",
            display_name="Load Images (Directory)",
            category="advanced/image",
            inputs=[
                io.String.Input(
                    "directory_path",
                    multiline=False,
                    placeholder="path/to/directory",
                    tooltip="Path to the directory containing images.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Index of the first image to load (sorted alphabetically)."
                ),
                io.Int.Input(
                    "load_count",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Number of images to load."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE", is_output_list=True),
                io.Mask.Output(display_name="MASK", is_output_list=True),
                io.Mask.Output(display_name="MASK_INVERTED", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, directory_path: str, start_index: int, load_count: int) -> io.NodeOutput:
        normalized_path = cls._normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        # Get valid image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mpo'}
        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(normalized_path, f))

        files.sort()

        # Apply slice
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        if not selected_files:
             raise ValueError(f"No images found in range [{start_index}:{end_index}] in directory: {directory_path}")

        output_images = []
        output_masks = []
        output_masks_inverted = []

        for file_path in selected_files:
            w, h = None, None
            try:
                img = node_helpers.pillow(Image.open, file_path)
            except Exception as e:
                print(f"Warning: Could not load image {file_path}: {e}")
                continue

            # Process just the first frame
            i = node_helpers.pillow(ImageOps.exif_transpose, img)

            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 65535))

            image = i.convert("RGB")

            if w is None:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                print(f"Warning: Skipping {file_path} due to dimension mismatch. Expected {w}x{h}, got {image.size}")
                continue

            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
                mask_inverted = 1.0 - mask
            elif i.mode == 'P' and 'transparency' in i.info:
                rgba = i.convert('RGBA')
                mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
                mask_inverted = 1.0 - mask
            else:
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
                mask_inverted = 1.0 - mask


            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))
            output_masks_inverted.append(mask_inverted.unsqueeze(0))

        if not output_images:
            raise ValueError("No valid images loaded (checked dimensions and validity).")


        return io.NodeOutput(output_images, output_masks, output_masks_inverted)

    @classmethod
    def IS_CHANGED(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = cls._normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mpo'}
        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort()
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                # Hash filename and mtime
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        return m.digest().hex()

class SwitchInverseNode(SwitchNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySwitchInverseNode",
            display_name="Switch (Inverse)",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_true", template=template, lazy=True),
                io.MatchType.Input("on_false", template=template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )


class SoftSwitchInverseNode(SoftSwitchNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySoftSwitchInverseNode",
            display_name="Soft Switch (Inverse)",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

class IntegerRangeRandom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntegerRangeRandom",
            display_name="Random Integer in Range",
            category="utils/primitive",
            inputs=[
                io.Int.Input("minimum", min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("maximum", min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("seed", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output(display_name="random_integer")],
        )

    @classmethod
    def execute(cls, minimum: int, maximum: int, seed: int = 0) -> io.NodeOutput:
        min_val = min(minimum, maximum)
        max_val = max(minimum, maximum)
        rng = random.Random(seed)
        return io.NodeOutput(rng.randint(min_val, max_val))


FLOW_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


class ImageMatchPropertiesNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageMatchProperties",
            display_name="Image Match Properties",
            category="advanced/image",
            inputs=[
                io.Image.Input("original_image"),
                io.Image.Input("generated_image"),
                io.Float.Input("overall_weight", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("color_weight", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("lighting_weight", default=1.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("texture_preservation", default=0.5, min=0.0, max=1.0, step=0.001, tooltip="Preserves edges and textures from the generated image by matching only low-frequency properties."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask to softly blend the color/lighting changes onto the generated image."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
        overall_weight: float,
        color_weight: float,
        lighting_weight: float,
        texture_preservation: float,
        mask: torch.Tensor = None,
    ) -> io.NodeOutput:
        result = _match_image_properties(
            original_image,
            generated_image,
            overall_weight,
            color_weight,
            lighting_weight,
            texture_preservation,
            mask,
        )
        return io.NodeOutput(result)


class OpticalFlowComposite(io.ComfyNode):
    """
    Composites a Klein edit onto the original image.

    v2.2: Global Rigid Alignment. Calculates a single global camera shift from
    unchanged background pixels and translates the entire generated image rigidly.
    Eliminates seam distortion while fixing AI background drift.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OpticalFlowComposite",
            display_name="Optical Flow Composite (Global Align)",
            category="advanced/image",
            inputs=[
                io.Image.Input("original_image"),
                io.Image.Input("generated_image"),
                io.Float.Input(
                    "delta_e_threshold",
                    default=-1.0, min=-1.0, max=100.0, step=1.0,
                    tooltip="How different a pixel's color must be to count as 'edited'. Higher values = only obvious edits are detected (smaller mask, more original preserved). Lower values = subtle changes are also captured (larger mask, more of the generated image used). Set to -1 for automatic tuning."
                ),
                io.Float.Input(
                    "grow_mask_pct",
                    default=0.0, min=-3.0, max=3.0, step=0.1,
                    tooltip="Expands or shrinks the detected edit region. Positive values grow the mask outward, capturing more of the surrounding area (useful if edges of the edit are being clipped). Negative values erode the mask inward, trimming the edges (useful if too much background is being pulled in)."
                ),
                io.Float.Input(
                    "feather_pct",
                    default=2.0, min=0.0, max=10.0, step=0.25,
                    tooltip="How gradually the edit blends into the original at the mask boundary. Higher values create a wider, softer transition (smoother blending, but may wash out fine edges). Lower values create a sharper, more abrupt cutover (crisper edges, but seams may be more visible)."
                ),
                io.Combo.Input(
                    "flow_quality",
                    options=["medium", "fast", "ultrafast"],
                    default="medium",
                    tooltip="Accuracy of the optical flow alignment between original and generated images. Higher quality = more precise change detection and alignment (slower). Lower quality = faster processing but may miss subtle shifts or produce noisier masks."
                ),
                io.Float.Input(
                    "occlusion_threshold",
                    default=-1.0, min=-1.0, max=20.0, step=0.5,
                    tooltip="Sensitivity to pixels that moved so much they can't be reliably matched between images. Higher values ignore more motion discrepancies (fewer false positives from camera jitter, but may miss real edits). Lower values flag more pixels as changed (catches more edits, but may over-detect in noisy areas). Set to -1 for automatic tuning."
                ),
                io.Float.Input(
                    "close_radius_pct",
                    default=0.5, min=0.0, max=5.0, step=0.1,
                    tooltip="Fills small holes and gaps inside the detected edit region. Higher values close larger gaps (creates a more solid, continuous mask). Lower values leave small holes intact (preserves finer mask detail but may leave speckled artifacts inside the edit)."
                ),
                io.Float.Input(
                    "min_region_pct",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Removes small isolated blobs from the mask that are likely false positives. Higher values filter out larger stray regions (cleaner mask, but may discard small intentional edits). Lower values keep smaller regions (preserves tiny edits, but may let through noise)."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="composited_image"),
                io.Mask.Output(display_name="change_mask"),
                io.String.Output(display_name="report"),
            ]
        )

    @classmethod
    def execute(cls, original_image, generated_image,
            delta_e_threshold=-1.0, grow_mask_pct=0.0, feather_pct=2.0,
            flow_quality="medium", occlusion_threshold=-1.0,
            close_radius_pct=0.5, min_region_pct=0.05):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        if orig_np.shape != gen_np.shape:
            H, W = gen_np.shape[:2]
            pil  = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        H, W = gen_np.shape[:2]
        diag = _diag(H, W)
        total_area = H * W

        grow_px    = round(grow_mask_pct * diag / 100.0)
        feather_px = abs(feather_pct) * diag / 100.0
        close_px   = _pct_to_px(close_radius_pct, diag)
        min_px     = max(0, round(min_region_pct * total_area / 100.0))

        result, change_mask, stats = _composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_px,
            close_radius        = close_px,
            min_region_px       = min_px,
            feather_px          = feather_px,
        )

        report_lines =[
            "=== Klein Edit Composite v2.2 (Global Align) ===",
            f"Resolution:       {stats['resolution']}  (diag {stats['diagonal_px']}px)",
            f"",
        ]

        if "auto_delta_e" in stats:
            report_lines.append(f"ΔE threshold:     AUTO → {stats['auto_delta_e']:.1f}")
        else:
            report_lines.append(f"ΔE threshold:     {delta_e_threshold:.1f}")

        if "auto_occlusion" in stats:
            report_lines.append(f"Occlusion thresh: AUTO → {stats['auto_occlusion']:.1f}")
        else:
            report_lines.append(f"Occlusion thresh: {occlusion_threshold:.1f}")

        report_lines +=[
            f"Grow mask:        {grow_mask_pct:+.1f}% → {grow_px:+d}px",
            f"Feather:          {feather_pct:.1f}% → {feather_px:.0f}px",
            f"Close radius:     {close_radius_pct:.1f}% → {close_px}px",
            f"Min region:       {min_region_pct:.2f}% → {min_px}px",
            f"Flow quality:     {flow_quality}",
            f"",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Occluded pixels:  {stats['occluded_px']:,}",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px",
            f"Flow p99 shift:   {stats['flow_p99_px']:.2f}px",
            f"Median ΔE:        {stats['median_de']:.2f}",
        ]

        return io.NodeOutput(torch.from_numpy(result).unsqueeze(0),
                torch.from_numpy(change_mask).unsqueeze(0),
                "\n".join(report_lines))


class TextOverlayNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextOverlayNode",
            display_name="Text Overlay",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("text", multiline=True, default="Hello World"),
                io.Int.Input("font_size", default=32, min=1, max=1024),
                io.String.Input("text_color", default="FFFFFF"),
                io.String.Input("bg_color", default="000000"),
                io.Boolean.Input("draw_background", default=True),
                io.Int.Input("bg_padding", default=10, min=0, max=1024),
                io.Float.Input("bg_transparency", default=0.5, min=0.0, max=1.0, step=0.05, tooltip="0.0 is fully transparent, 1.0 is fully opaque"),
                io.Boolean.Input("use_percentage", default=False, tooltip="If True, top/bottom/left/right are treated as percentages (0-100) of the image size."),
                io.Int.Input("top", default=-1, min=-1, max=8192, tooltip="-1 for center vertically or use bottom offset"),
                io.Int.Input("bottom", default=-1, min=-1, max=8192, tooltip="-1 for center vertically or use top offset"),
                io.Int.Input("left", default=-1, min=-1, max=8192, tooltip="-1 for center horizontally or use right offset"),
                io.Int.Input("right", default=-1, min=-1, max=8192, tooltip="-1 for center horizontally or use left offset"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        text: str,
        font_size: int,
        text_color: str,
        bg_color: str,
        draw_background: bool,
        bg_padding: int,
        bg_transparency: float,
        use_percentage: bool,
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> io.NodeOutput:

        t_color = _hex_to_rgb(text_color, (255, 255, 255))

        # Calculate background color with transparency (alpha 0-255)
        b_color_base = _hex_to_rgb(bg_color, (0, 0, 0))
        alpha = int(bg_transparency * 255.0)
        # Ensure b_color is exactly 4 elements long for RGBA
        if len(b_color_base) == 3:
            b_color = (b_color_base[0], b_color_base[1], b_color_base[2], alpha)
        else: # Handle case where _hex_to_rgb returns a 4-element tuple or default
            b_color = (b_color_base[0], b_color_base[1], b_color_base[2], alpha)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Linux fallback
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        # Handle batch of images
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        output_images = []

        for i in range(batch_count):
            img_tensor = image[i] if batch_count > 1 else image
            # Tensor is typically (C, H, W) or (H, W, C) depending on context, assuming (H, W, C) here
            # Convert to PIL
            img_pil = Image.fromarray(np.clip(255.0 * img_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGBA")

            draw = ImageDraw.Draw(img_pil)

            # Calculate text size using textbbox
            left_box, top_box, right_box, bottom_box = draw.textbbox((0, 0), text, font=font)
            text_width = right_box - left_box
            text_height = bottom_box - top_box

            img_width, img_height = img_pil.size

            # Calculate total width/height including background padding
            total_width = text_width + (bg_padding * 2 if draw_background else 0)
            total_height = text_height + (bg_padding * 2 if draw_background else 0)

            # Resolve coordinates based on mode (pixels vs percentage)
            def resolve_coord(val, max_val):
                if val == -1:
                    return -1
                if use_percentage:
                    return int((val / 100.0) * max_val)
                return val

            l_resolved = resolve_coord(left, img_width)
            r_resolved = resolve_coord(right, img_width)
            t_resolved = resolve_coord(top, img_height)
            b_resolved = resolve_coord(bottom, img_height)

            # Determine X position
            if l_resolved == -1 and r_resolved == -1:
                x_pos = (img_width - total_width) // 2
            elif l_resolved != -1:
                x_pos = l_resolved
            else: # r_resolved != -1
                x_pos = img_width - total_width - r_resolved

            # Determine Y position
            if t_resolved == -1 and b_resolved == -1:
                y_pos = (img_height - total_height) // 2
            elif t_resolved != -1:
                y_pos = t_resolved
            else: # b_resolved != -1
                y_pos = img_height - total_height - b_resolved

            # Draw background
            if draw_background:
                bg_rect = [x_pos, y_pos, x_pos + total_width, y_pos + total_height]
                # To support alpha, we draw on a separate layer and composite
                overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(bg_rect, fill=b_color)
                img_pil = Image.alpha_composite(img_pil, overlay)
                draw = ImageDraw.Draw(img_pil) # Re-init draw for text on composited image

            # Draw text
            text_x = x_pos + (bg_padding if draw_background else 0)
            text_y = y_pos + (bg_padding if draw_background else 0)

            # Use textbbox offset for more accurate vertical alignment of text
            draw.text((text_x - left_box, text_y - top_box), text, fill=t_color, font=font)

            # Convert back to tensor (RGB)
            img_pil = img_pil.convert("RGB")
            out_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)
            output_images.append(out_tensor)

        if batch_count > 1:
            out = torch.stack(output_images, dim=0)
        else:
            out = output_images[0].unsqueeze(0)

        return io.NodeOutput(out)

class TagNormalizeCombine(io.ComfyNode):
    """
    Node that normalizes scores in two sets of tags and combines them,
    deduplicating and sorting by the normalized scores.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TagNormalizeCombine",
            display_name="Tag Normalize and Combine",
            category="advanced/text",
            inputs=[
                io.String.Input("tags_1", multiline=True, default=""),
                io.String.Input("tags_2", multiline=True, default=""),
                io.AnyType.Input(
                    "scores_1",
                    tooltip="Dictionary of scores for tags_1",
                    optional=True,
                ),
                io.AnyType.Input(
                    "scores_2",
                    tooltip="Dictionary of scores for tags_2",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="deduped_tags"),
                io.AnyType.Output(display_name="normalized_scores"),
            ],
        )

    @staticmethod
    def normalize_scores(scores: dict, min_val=0.000001, max_val=0.999999) -> dict:
        if not scores:
            return {}

        current_scores = [float(v) for v in scores.values()]
        current_min = min(current_scores)
        current_max = max(current_scores)

        if current_max == current_min:
            return {k: max_val for k in scores.keys()}

        normalized = {}
        for k, v in scores.items():
            norm = min_val + (float(v) - current_min) / (current_max - current_min) * (
                max_val - min_val
            )
            normalized[k] = norm
        return normalized

    @classmethod
    def execute(
        cls, tags_1: Union[str, list], tags_2: Union[str, list], scores_1: Union[str, dict] = None, scores_2: Union[str, dict] = None
    ) -> io.NodeOutput:
        # Parse tags
        def parse_tags(t_input):
            if isinstance(t_input, list):
                return [str(t).strip() for t in t_input if t]
            if not t_input or not isinstance(t_input, str):
                return []
            # Split by comma and handle potential spaces
            return [t.strip() for t in t_input.split(",") if t.strip()]

        t1_list = parse_tags(tags_1)
        t2_list = parse_tags(tags_2)

        # Handle scores
        def process_scores(s_input, t_list):
            if s_input is None:
                # Generate even distribution
                if not t_list:
                    return {}
                num_tags = len(t_list)
                if num_tags == 1:
                    return {t_list[0]: 0.999999}

                max_v = 0.999999
                min_v = 0.000001
                scores = {}
                for i, tag in enumerate(t_list):
                    # Linearly interpolate from max_v down to min_v
                    score = max_v - i * (max_v - min_v) / (num_tags - 1)
                    scores[tag] = score
                return scores

            # Parse existing scores
            def parse_s(s_in):
                if isinstance(s_in, dict):
                    return s_in
                if not s_in or not isinstance(s_in, str):
                    return {}
                try:
                    return json.loads(s_in)
                except json.JSONDecodeError:
                    return {}

            return cls.normalize_scores(parse_s(s_input))

        norm_s1 = process_scores(scores_1, t1_list)
        norm_s2 = process_scores(scores_2, t2_list)

        # Combine and deduplicate
        combined_scores = {}

        # Process first set
        for t in t1_list:
            score = norm_s1.get(t, 0.000001)
            combined_scores[t] = score

        # Process second set with deduplication logic
        for t in t2_list:
            score = norm_s2.get(t, 0.000001)
            if t in combined_scores:
                # Keep the one with the highest normalized score
                if score > combined_scores[t]:
                    combined_scores[t] = score
            else:
                combined_scores[t] = score

        # Sort tags by normalized scores (descending)
        sorted_tags = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Prepare outputs
        deduped_tags_str = ", ".join(sorted_tags)
        normalized_scores_dict = {tag: combined_scores[tag] for tag in sorted_tags}

        return io.NodeOutput(deduped_tags_str, normalized_scores_dict)


class RandInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PrimitiveRandomInt",
            display_name="RandomInt",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class StaticInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PrimitiveStaticInt",
            display_name="StaticInt",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class RandIntRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PrimitiveRandomIntRange",
            display_name="RandomIntRange",
            category="utils/primitive",
            inputs=[
                io.Int.Input("min", default=0, min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("max", default=100, min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, min: int, max: int, seed: int) -> io.NodeOutput:
        rng = random.Random(seed)
        return io.NodeOutput(rng.randint(min, max))

class UNETLoaderAsync(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UNETLoaderAsync",
            display_name="Load Diffusion Model (Async)",
            category="advanced/model",
            inputs=[
                io.Combo.Input("unet_name", folder_paths.get_filename_list("diffusion_models"), tooltip="Select a UNET model to load."),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], advanced=True),
                io.Boolean.Input("disable_dynamic_vram", default=False, advanced=True, tooltip="If true, disables dynamic VRAM optimizations when loading the UNET. This can reduce VRAM usage at the cost of potentially higher CPU usage and slower performance. Recommended to keep enabled unless you are experiencing VRAM-related issues with certain models.")
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, unet_name: str, weight_dtype: str, disable_dynamic_vram: bool) -> io.NodeOutput:
         # Placeholder for actual UNET loading logic
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = load_diffusion_model(unet_path, model_options=model_options, disable_dynamic=disable_dynamic_vram)
        return io.NodeOutput(model)


class TextGenerateQwen35SystemPrompt(io.ComfyNode):
    """
    TextGenerate variant for Qwen3.5 models with custom system message support.
    Builds the chat template via string concatenation (never .format()) so any
    characters in user input — including { } \\ and control sequences — are safe.
    """

    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="TextGenerateQwen35SystemPrompt",
            display_name="Text Generate Qwen3.5 (System Prompt)",
            category="advanced/textgen",
            search_aliases=["LLM", "VLM", "qwen", "qwen35", "system prompt", "textgen"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="",
                    tooltip="User message. All characters including { } are safe — template uses concatenation not .format()."),
                io.String.Input("system_message", multiline=True, dynamic_prompts=False, default="",
                    tooltip="System message injected before the user turn. Leave empty to skip the system block entirely."),
                io.Image.Input("image", optional=True),
                io.Int.Input("max_length", default=512, min=1, max=8192),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False,
                    tooltip="Enable thinking mode. When False, suppresses thinking with <think>\\n</think>\\n."),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
            ],
        )

    @classmethod
    def _build_prompt(cls, prompt: str, system_message: str, has_image: bool, thinking: bool) -> str:
        """
        Build the full Qwen3.5 chat-template string using concatenation only.
        No .format() / %-formatting — user-supplied strings are never interpolated,
        so { } and any other characters are completely safe.

        Template structure (matches qwen35.py Qwen35ImageTokenizer):
            [<|im_start|>system\\n{system_message}<|im_end|>\\n]   <- optional
            <|im_start|>user\\n
            [<|vision_start|><|image_pad|><|vision_end|>]          <- if image
            {prompt}<|im_end|>\\n
            <|im_start|>assistant\\n
            [<think>\\n</think>\\n]                                  <- if not thinking
        """
        result = ""

        # Optional system block
        if system_message and system_message.strip():
            result = (
                "<|im_start|>system\n"
                + system_message
                + "<|im_end|>\n"
            )

        # User block — image token placed before text per qwen35.py:761
        result += "<|im_start|>user\n"
        if has_image:
            result += "<|vision_start|><|image_pad|><|vision_end|>"
        result += prompt + "<|im_end|>\n"

        # Assistant block
        result += "<|im_start|>assistant\n"

        # Suppress thinking unless thinking mode requested (matches qwen35.py:784-785)
        if not thinking:
            result += "<think>\n</think>\n"

        return result

    @classmethod
    def execute(cls, clip, prompt, system_message, max_length, sampling_mode,
                image=None, thinking=False) -> io.NodeOutput:

        formatted_prompt = cls._build_prompt(
            prompt=prompt,
            system_message=system_message,
            has_image=image is not None,
            thinking=thinking,
        )

        # skip_template=True because we built the full template ourselves.
        # The tokenizer detects <|im_start|> prefix and skips its own template (qwen35.py:769).
        tokens = clip.tokenize(
            formatted_prompt,
            image=image,
            skip_template=True,
            min_length=1,
            thinking=thinking,
        )

        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        generated_text = clip.decode(generated_ids, skip_special_tokens=True)
        return io.NodeOutput(generated_text)


class ColorConvertNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorConvertNode",
            display_name="Color Convert (Bottom of node is color selector)",
            category="advanced/color",
            inputs=[
                io.Combo.Input("from_mode", options=["auto", "Manual Hex #FFFFFF", "Int 0-16777215", "Comma separated 255,255,255"], default="auto", tooltip="Select how to interpret the color input. 'Auto' will use the color picker input unless one of the other fields is filled in, in which case it will use the filled-in field based on the other options. The other modes will take precedence over the color picker input when their respective fields are filled in."),
                io.Color.Input("color_hex", default="#FFFFFF", display_name="This is not the Color Selector →", tooltip="Select a color using the color picker. This input is used when 'from_mode' is set to 'auto' and no other manual inputs are provided. The selected color will be converted to hex, int, and string formats for output."),
                io.String.Input("manual_hex", multiline=False, optional=True, display_name="Manual Hex Input", tooltip="Enter a hex color code manually, e.g. #FF00FF. Takes precedence over the color picker input if 'from_mode' is set to 'Manual Hex #FFFFFF'."),
                io.Int.Input("color_int", min=-1, max=16777215, default=-1, optional=True, display_name="Color Int Input", tooltip="Enter a color as an integer (0-16777215). Interpreted as 0xRRGGBB. Takes precedence over the color picker input if 'from_mode' is set to 'Int 0-16777215'."),
                io.String.Input("comma_separated", multiline=False, optional=True, display_name="Comma Separated Input", tooltip="Enter a color as comma-separated RGB values (0-255), e.g. 255,0,255. Takes precedence over the color picker input if 'from_mode' is set to 'Comma separated 255,255,255'."),
            ],
            outputs=[
                io.String.Output(display_name="color_hex_output"),
                io.Int.Output(display_name="color_int_output"),
                io.String.Output(display_name="color_string_output"),
            ]
        )

    @classmethod
    def _validate_hex(cls, s):
        """Return (r, g, b) if s is a valid 6-digit hex string (case-insensitive, # prefix optional), else None."""
        if s is None:
            return None
        s = s.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            return None
        if not all(c in "0123456789abcdefABCDEF" for c in s):
            return None
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)

    @classmethod
    def _validate_int(cls, n):
        """Return (r, g, b) if n is an int in 0–16777215, else None."""
        if n is None or not isinstance(n, int):
            return None
        if not (0 <= n <= 16777215):
            return None
        return ((n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)

    @classmethod
    def _validate_csv(cls, s):
        """Return (r, g, b) if s is 3 ints 0–255 separated by ',' or ', ', else None."""
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            return None
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None
        if not all(0 <= v <= 255 for v in vals):
            return None
        return tuple(vals)

    @classmethod
    def _rgb_to_outputs(cls, r, g, b):
        return (
            f"#{r:02X}{g:02X}{b:02X}",
            (r << 16) | (g << 8) | b,
            f"{r}, {g}, {b}",
        )

    @classmethod
    def execute(cls, from_mode, color_hex, manual_hex=None, color_int=None, comma_separated=None) -> io.NodeOutput:
        # Sanitize sentinels to None
        if color_int is None or not (0 <= color_int <= 16777215):
            color_int = None
        if not manual_hex or manual_hex.strip() == "":
            manual_hex = None
        if not comma_separated or comma_separated.strip() == "":
            comma_separated = None

        if from_mode == "auto":
            valid_hex = cls._validate_hex(manual_hex)
            valid_int = cls._validate_int(color_int)
            valid_csv = cls._validate_csv(comma_separated)
            valid_inputs = [x for x in [valid_hex, valid_int, valid_csv] if x is not None]

            if len(valid_inputs) == 1:
                r, g, b = valid_inputs[0]
            else:
                if len(valid_inputs) > 1:
                    print("[ColorConvertNode] Warning: multiple optional inputs are valid in auto mode, falling back to color picker.")
                # Use picker (len==0 or len>1)
                picker = cls._validate_hex(color_hex)
                if picker is None:
                    raise ValueError(f"Color picker value '{color_hex}' must be in format #RRGGBB with valid hex digits")
                r, g, b = picker

            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Manual Hex #FFFFFF":
            result = cls._validate_hex(manual_hex)
            if result is None:
                raise ValueError(f"Manual hex input '{manual_hex}' must be 6 valid hex digits (0-9, a-f, A-F), with optional # prefix")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Int 0-16777215":
            result = cls._validate_int(color_int)
            if result is None:
                raise ValueError(f"Color integer input '{color_int}' must be in range 0–16777215")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Comma separated 255,255,255":
            result = cls._validate_csv(comma_separated)
            if result is None:
                raise ValueError(f"Comma-separated input '{comma_separated}' must be 3 integers 0–255 separated by ',' or ', '")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        else:
            raise ValueError(f"Unknown from_mode '{from_mode}'")

        return io.NodeOutput(color_hex_output, color_int_output, color_string_output)


_LORA_LOADER_CACHE = None


class LoraLoaderCLIPOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoraLoaderCLIPOnly",
            display_name="Load LoRA for CLIP Only",
            category="advanced/model",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("lora_name", folder_paths.get_filename_list("loras"), tooltip="Select a LoRA model to load. This node will attempt to extract and load only the CLIP portion of the LoRA, which can be useful for certain text/image embedding applications. Note that not all LoRA models will have a CLIP portion, and results may vary depending on the model architecture."),
                io.Float.Input("strength_clip", default=1.0, min=0.0, max=1.0, tooltip="The strength of the CLIP portion to apply."),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, clip, lora_name: str, strength_clip: float) -> io.NodeOutput:
        global _LORA_LOADER_CACHE
        if strength_clip == 0:
            return (io.NodeOutput(clip),)
        # Placeholder for actual LoRA loading logic
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if _LORA_LOADER_CACHE is not None:
            if _LORA_LOADER_CACHE[0] == lora_path:
                lora = _LORA_LOADER_CACHE[1]
            else:
                _LORA_LOADER_CACHE = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            _LORA_LOADER_CACHE = (lora_path, lora)

        clip_lora = comfy.sd.load_lora_for_models(None, clip, lora, 0, strength_clip)[1]
        return io.NodeOutput(clip_lora)

class SamplingUtils(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LlamaTokenizerOptions,
            SamplingParameters,
            AdjustedResolutionParameters,
            GetJsonKeyValue,
            Image_Color_Noise,
            TextEncodeFlux2SystemPrompt,
            TextEncodeKleinSystemPrompt,
            TextEncodeLtxv2SystemPrompt,
            TextEncodeZITSystemPrompt,
            TextEncodeZImageThinkPrompt,
            TextEncodeSystemPrompt,
            ModifyMask,
            ImageBlendByMask,
            SU_InjectNoiseToLatent,
            SystemMessagePresets,
            InstructPromptPresets,
            BonusPromptPresets,
            EditTargetPresets,
            EditOpPresets,
            CameraShotPresets,
            VLMSysInstrPresets,
            VLMSysQueryAddPresets,
            VLMSysInstrAdvPresets,
            UnifiedPresets,
            FrakturPadNode,
            UnFrakturPadNode,
            JoinerPadding,
            IdeographicTagPad,
            IdeographicLinePad,
            IdeographicSentencePad,
            TagNormalizeCombine,
            SU_LoadImagePath,
            SU_LoadImageDirectory,
            SwitchInverseNode,
            SoftSwitchInverseNode,
            IntegerRangeRandom,
            ImageMatchPropertiesNode,
            OpticalFlowComposite,
            TextOverlayNode,
            RandInt,
            StaticInt,
            RandIntRange,
            UNETLoaderAsync,
            TextGenerateQwen35SystemPrompt,
            ColorConvertNode,
            LoraLoaderCLIPOnly,
        ]


async def comfy_entrypoint() -> SamplingUtils:
    return SamplingUtils()
