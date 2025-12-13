import sys
import json
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from typing import Union, List
import torch
import numpy as np
import scipy
import hashlib
import pilgram
from tqdm import tqdm
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy import model_management
import nodes
import node_helpers
from nodes import MAX_RESOLUTION
from .system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I



def round_to_nearest(n, m):
    return int((n + (m / 2)) // m) * m



class SamplingParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplingParameters",
            category="utils",
            inputs=[
                io.Int.Input(id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
                io.Float.Input(id="scale_by", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="How much to upscale initial resolution by for the upscaled one."),
                io.Int.Input(id="multiple", default=16, min=4, max=128, step=4, tooltip="Nearest multiple of the result to set the upscaled resolution to."),
                io.Int.Input(id="steps", default=26, min=1, max=10000, step=1, tooltip="How many steps to run the sampling for."),
                io.Float.Input(id="cfg", default=3.5, min=-100.0, max=100.0, step=0.01, tooltip="The amount of influence your prompot will have on the final image."),
                io.Int.Input(id="seed", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
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
    def execute(cls, *, width: int, height: int, batch_size: int=1, scale_by: float, multiple: int, steps: int, cfg: float, seed: int) -> io.NodeOutput:
        upscaled_width = round_to_nearest(int(width*scale_by), int(multiple))
        upscaled_height = round_to_nearest(int(height*scale_by), int(multiple))
        if scale_by > 2.0:
            tile_width = round_to_nearest(int((upscaled_width - (width / scale_by)) / scale_by), int(multiple))
            tile_height = round_to_nearest(int((upscaled_height - (height / scale_by)) / scale_by), int(multiple))
            tile_padding = round_to_nearest(int(max(width, height) - max(tile_width, tile_height)), int(multiple))
        else:
            tile_width = round_to_nearest(int(upscaled_width * 0.5), int(multiple))
            tile_height = round_to_nearest(int(upscaled_height * 0.5), int(multiple))
            tile_padding = round_to_nearest(int(max(width, height) - max(tile_width, tile_height)), int(multiple))
        width = round_to_nearest(int(width), int(multiple))
        height = round_to_nearest(int(height), int(multiple))
        return io.NodeOutput(width, height, batch_size, upscaled_width, upscaled_height, steps, cfg, seed, tile_width, tile_height, tile_padding)

class GetJsonKeyValue(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetJsonKeyValue",
            category="utils",
            inputs=[
                io.String.Input("json_path", default="./input/JSON_KeyValueStore.json", multiline=False, tooltip="Path to a .json file with simple top level structure with key and value. See example in custom node folder."),
                io.Combo.Input("key_id_method", options=["custom", "random_rotate", "increment_rotate"]),
                io.Int.Input("rotation_interval", default=0, tooltip="how many steps to jump when doing rotate."),
                io.String.Input("key_id", default="placeholder", multiline=False, tooltip="Put name of key in the .json here if using custom in key_id_method."),
            ],
            outputs=[
                io.String.Output(display_name="key_value")
            ],
        )

    @classmethod
    def execute(cls, json_path, key_id_method, rotation_interval, key_id="placeholder") -> io.NodeOutput:
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
            with open(absolute_json_path, 'r') as f:
                api_keys_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"RotateKeyAPI Error: JSON file not found at {absolute_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"RotateKeyAPI Error: Could not decode JSON from {absolute_json_path}. Check file format.")
        except Exception as e:
            raise RuntimeError(f"RotateKeyAPI Error: Unexpected error reading file {absolute_json_path}: {e}")

        if not isinstance(api_keys_data, dict):
            raise ValueError(f"RotateKeyAPI Error: JSON content is not a dictionary in {absolute_json_path}. Expected format: {{'key_id': 'api_key', ...}}")

        if not api_keys_data:
             raise ValueError(f"RotateKeyAPI Error: The JSON dictionary in {absolute_json_path} is empty.")

        selected_key_value = None

        if key_id_method == "custom":
            if key_id == "placeholder":
                 print("RotateKeyAPI Warning: 'custom' method selected but 'key_id' is still the default 'placeholder'. Ensure this is intended or provide a valid key ID.")

            selected_key_value = api_keys_data.get(key_id)

            if selected_key_value is None:
                 raise ValueError(f"RotateKeyAPI Error: Custom key ID '{key_id}' not found in the JSON dictionary keys.")


        elif key_id_method == "random_rotate":
            api_keys_list = list(api_keys_data.values())

            selected_key_value = random.choice(api_keys_list)

        elif key_id_method == "increment_rotate":
             api_keys_list = list(api_keys_data.values())

             index = rotation_interval % len(api_keys_list)

             try:
                selected_key_value = api_keys_list[index]
             except IndexError:
                 raise IndexError(f"RotateKeyAPI Error: Calculated index {index} (from interval {rotation_interval}) is out of bounds for list of size {len(api_keys_list)}.")
             except Exception as e:
                  raise RuntimeError(f"RotateKeyAPI Error: Unexpected error accessing item at index {index}: {e}")

        if not isinstance(selected_key_value, str) or not selected_key_value:
             raise ValueError(f"RotateKeyAPI Error: Retrieved value for selected key is not a valid string. Value: {selected_key_value}")


        print(f"RotateKeyAPI: Successfully retrieved API key using method '{key_id_method}'.")
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
                io.Float.Input("attenuation", default=0.5, max=100.0, min=0.0, step=0.01),
                io.Combo.Input("noise_type", options=["grey", "white", "red", "pink", "green", "blue", "mix"]),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                io.Image.Output(display_name="noise_image"),
            ],
        )

    @classmethod
    def execute(cls, width, height, frequency, attenuation, noise_type, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise_image = cls.generate_power_noise(width, height, frequency, attenuation, noise_type, generator)
        return io.NodeOutput(pil2tensor(noise_image))

    @classmethod
    def generate_power_noise(cls, width, height, frequency, attenuation, noise_type, generator):

        def normalize_array(arr):
            return (255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))).astype(np.uint8)

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
            power_spectrum = f ** power_modifier
            fft_noise = np.fft.fft2(noise)
            fft_modified = fft_noise * power_spectrum
            inv_fft = np.fft.ifft2(fft_modified)
            return np.real(inv_fft)

        noise_array = np.zeros((height, width, 3), dtype=np.uint8)
        zeros_channel = np.zeros((height, width), dtype=np.uint8)

        if noise_type == "grey":
            luma = normalize_array(grey_noise_texture(width, height, attenuation, generator))
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
            r = normalize_array(fourier_noise(width, height, attenuation, -1.0, generator)) # Pink Frequency
            g = normalize_array(fourier_noise(width, height, attenuation, 0.5, generator))  # Green Frequency
            b = normalize_array(fourier_noise(width, height, attenuation, 1.0, generator))   # Blue Frequency
            noise_array = np.stack([r, g, b], axis=-1)

        else:
            cstr(f"Unsupported noise type `{noise_type}`").error.print()
            return Image.new("RGB", (width, height), color="black")

        return Image.fromarray(noise_array, 'RGB')

class SystemMessagePresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemMessagePresets",
            category="advanced/conditioning",
            inputs=[
                io.Combo.Input("preset", options=[
                    "F2_SYSTEM_MESSAGE",
                    "F2_SYSTEM_MESSAGE_UPSAMPLING_I2I",
                    "F2_SYSTEM_MESSAGE_UPSAMPLING_T2I"
                ], default="F2_SYSTEM_MESSAGE"),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = {
            "F2_SYSTEM_MESSAGE": SYSTEM_MESSAGE,
            "F2_SYSTEM_MESSAGE_UPSAMPLING_I2I": SYSTEM_MESSAGE_UPSAMPLING_I2I,
            "F2_SYSTEM_MESSAGE_UPSAMPLING_T2I": SYSTEM_MESSAGE_UPSAMPLING_T2I,
        }
        system_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(system_prompt)

class TextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
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

class TextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeZITSystemPrompt",
            category="advanced/conditioning",
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
            template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
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
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("expand", default=0, max=MAX_RESOLUTION, min=-MAX_RESOLUTION, step=1),
                io.Float.Input("incremental_expandrate", default=0.0, max=100.0, min=0.0, step=0.01),
                io.Boolean.Input("tapered_corners", default=True),
                io.Boolean.Input("flip_input", default=False),
                io.Float.Input("blur_radius", default=0.0, max=100.0, min=0.0, step=0.01),
                io.Float.Input("lerp_alpha", default=1.0, max=1.0, min=0.0, step=0.01),
                io.Float.Input("decay_factor", default=1.0, max=1.0, min=0.0, step=0.01),
                io.Boolean.Input("fill_holes", default=False, optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Mask.Output(display_name="mask_inverted"),
            ],
        )

    @classmethod
    def execute(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor, fill_holes=False):
        import kornia.morphology as morph

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

        alpha = lerp_alpha
        decay = decay_factor

        # 1. Clone the original mask to keep a reference to the un-blurred pixels
        original_mask_input = mask.clone()

        if flip_input:
            mask = 1.0 - mask
            original_mask_input = 1.0 - original_mask_input

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))

        # Prepare original mask for processing loop (match dimensions)
        original_mask_batches = original_mask_input.reshape((-1, mask.shape[-2], mask.shape[-1]))

        out = []
        previous_output = None
        current_expand = expand
        for m in tqdm(growmask, desc="Expanding/Contracting Mask"):
            output = m.unsqueeze(0).unsqueeze(0).to(model_management.get_torch_device())  # Add batch and channel dims for kornia
            if abs(round(current_expand)) > 0:
                # Create kernel - kornia expects kernel on same device as input
                if tapered_corners:
                    kernel = torch.tensor([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]], dtype=torch.float32, device=output.device)
                else:
                    kernel = torch.tensor([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]], dtype=torch.float32, device=output.device)

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
            mask = blurred
            mask_inverted = 1.0 - blurred
            return io.NodeOutput(mask, mask_inverted)
        else:
            mask = torch.stack(out, dim=0)
            mask_inverted = 1.0 - torch.stack(out, dim=0)
            return io.NodeOutput(mask, mask_inverted)


class ImageBlendByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageBlendByMask",
            category="utils/mask",
            inputs=[
                io.Image.Input("destination"),
                io.Image.Input("source"),
                io.Combo.Input("mode", options=[
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
                    "soft_light"
                ], default="add"),
                io.Float.Input("blend_percentage", default=1.0, max=1.0, min=0.0, step=0.01),
                io.Boolean.Input("resize_source", default=False),
                io.Mask.Input("mask"),
            ],
            outputs=[
                io.Image.Output(display_name="blended_image"),
            ],
        )

    @classmethod
    def execute(self, destination, source, mode='add', blend_percentage=1.0, resize_source=False, mask=None):

        # Tensor to PIL
        def tensor2pil(image):
            return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        # PIL to Tensor
        def pil2tensor(image):
            return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        source = source.movedim(-1, 1).to(destination.device)

        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[-2], destination.shape[-1]), mode="bicubic")

        # Convert images to PIL
        img_a = tensor2pil(destination)
        img_b = tensor2pil(source)

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
            "soft_light": pilgram.css.blending.soft_light
        }

        out_image = blending_modes.get(mode, pilgram.css.blending.normal)(img_a, img_b)

        out_image = out_image.convert("RGB")

        # Apply mask if provided
        if mask is not None:
            mask = ImageOps.invert(tensor2pil(mask).convert('L'))
            out_image = Image.composite(img_a, out_image, mask.resize(img_a.size))

        # Blend image based on blend percentage
        blend_mask = Image.new(mode="L", size=img_a.size, color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(img_a, out_image, blend_mask)

        blended_image = pil2tensor(out_image)
        return io.NodeOutput(blended_image)


class SamplingUtils(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SamplingParameters,
            GetJsonKeyValue,
            Image_Color_Noise,
            TextEncodeFlux2SystemPrompt,
            TextEncodeZITSystemPrompt,
            ModifyMask,
            ImageBlendByMask,
            SystemMessagePresets,
        ]


async def comfy_entrypoint() -> SamplingUtils:
    return SamplingUtils()
