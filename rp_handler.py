from datetime import datetime

print(f"SETUP ---- A {datetime.now()}");

import os

import runpod
import io
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schema import INPUT_SCHEMA

print(f"SETUP ---- B {datetime.now()}");


os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import base64

print(f"SETUP ---- C {datetime.now()}");

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

print(f"SETUP ---- D {datetime.now()}");


def process(job):
    print(f"RUN ---- A {datetime.now()}");

    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    print(f"RUN ---- B {datetime.now()}");

    job_image_str = job_input['image']
    job_image_bytes = base64.b64decode(job_image_str)
    job_image = Image.open(io.BytesIO(job_image_bytes))

    if validated_input['seed'] is None:
        validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

    if validated_input['simplify'] is None:
        validated_input['simplify'] = 0.95

    if validated_input['texture_size'] is None:
        validated_input['texture_size'] = 1024

    print(f"RUN ---- C {datetime.now()}");

    # Run the pipeline
    outputs = pipeline.run(
        job_image,
        seed=validated_input['seed'],
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    print(f"RUN ---- D {datetime.now()}");

    # GLB files can be extracted from the outputs
    out_file = f"{job['id']}.glb"

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=validated_input['simplify'],          # Ratio of triangles to remove in the simplification process
        texture_size=validated_input['texture_size'],      # Size of the texture used for the GLB
    )

    glb.export(out_file)

    print(f"RUN ---- E {datetime.now()}");

    with open(out_file, "rb") as f:
       out_data = f.read()

    out_b64 = base64.b64encode(out_data).decode("utf-8")

    job_output = {
        "glb": out_b64
    }

    print(f"RUN ---- F {datetime.now()}");

    return job_output

def run(job):
    print(f"RUN ---- START {datetime.now()}");

    result = process(job)
    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    print(f"RUN ---- END {datetime.now()}");

    return result


if __name__ == '__main__':
	runpod.serverless.start({"handler": run})
