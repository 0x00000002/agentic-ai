# Image Generation Tool (`generate_image`)

This internal tool allows the agent framework to generate images based on text descriptions using Stability AI's Stable Diffusion 3.5 Medium model via the Replicate API.

## Configuration (`src/config/tools.yml`)

```yaml
- name: "generate_image"
  description: "Generates an image from a text description using the latest Stable Diffusion 3.5 Medium model via Replicate API. Use this tool when asked to create, draw, or generate an image."
  module: "src.tools.media.image_generator"
  function: "generate_image"
  parameters_schema:
    type: "object"
    properties:
      prompt:
        type: "string"
        description: "Description of the image to generate"
      negative_prompt:
        type: "string"
        description: "Elements to avoid in the generated image"
      width:
        type: "integer"
        description: "Image width (used to approximate aspect_ratio if aspect_ratio kwarg not provided)"
        default: 1024
      height:
        type: "integer"
        description: "Image height (used to approximate aspect_ratio if aspect_ratio kwarg not provided)"
        default: 1024
      num_inference_steps:
        type: "integer"
        description: "Number of denoising steps (passed as 'steps' to API)"
        default: 40 # Default for SD 3.5 Medium
      # Optional kwargs like aspect_ratio, output_format, output_quality, cfg can be passed via **kwargs
    required: ["prompt"]
  category: "media"
  source: "internal"
  speed: "medium"
  safety: "external"
```

## Implementation (`src/tools/media/image_generator.py`)

- The tool uses the model-specific Replicate API endpoint for `stability-ai/stable-diffusion-3.5-medium`, which runs the latest available version of that model.
- It requires the `REPLICATE_API_TOKEN` environment variable to be set for authentication.
- The function accepts additional `**kwargs` which are passed directly to the Replicate API's `input` payload. This allows using model-specific parameters like `aspect_ratio`, `output_format`, `output_quality`, `cfg`, etc.
- If `aspect_ratio` is not provided via `kwargs`, the function attempts to approximate a common aspect ratio (e.g., `1:1`, `16:9`) based on the `width` and `height` parameters. Providing `aspect_ratio` directly is recommended for SD 3.5 Medium.
- The underlying function implementing the API call has its own internal timeout for polling Replicate results, which is longer than the default tool execution timeout.

## Coordinator Integration

Currently, the `Coordinator` agent uses the `RequestAnalyzer` to detect image generation intent based on keywords. If detected, the `Coordinator` bypasses the standard LLM agent flow and directly calls this `generate_image` tool via the `ToolManager` to ensure reliability.
