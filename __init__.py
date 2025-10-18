import sys
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes

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
        return io.NodeOutput(width, height, batch_size, upscaled_width, upscaled_height, steps, cfg, seed, tile_width, tile_height, tile_padding)

class SamplingUtils(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SamplingParameters,
        ]


async def comfy_entrypoint() -> SamplingUtils:
    return SamplingUtils()
