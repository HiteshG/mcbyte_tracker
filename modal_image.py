import modal

print("Building image...")

image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install(
        "build-essential",      # gcc, g++, make (netifaces, thinplate)
        "git",
        "curl",
        "wget",
        "libgl1",               # OpenCV
        "libglib2.0-0",         # OpenCV runtime
    )
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "boxmot",
        "opencv-python-headless",
        "pandas",
        "scikit-learn",
        "transformers",
        "umap-learn",
        "pillow",
        "tqdm",
        "ipywidgets",
        "matplotlib",
        "numpy>=1.23.0,<2.0.0",
        "opencv-python>=4.6.0",
        "scipy>=1.9.0",
        "lap>=0.4.0",
        "omegaconf>=2.2.0",
        "hydra-core>=1.2.0",
        "charset-normalizer",   # replaces cchardet
    )
    .pip_install("git+https://github.com/facebookresearch/sam2.git")
    .pip_install(
        "git+https://github.com/hkchengrex/Cutie.git", 
        extra_options="--no-deps"           # avoid cchardet
    )
)

# Configuration
app = modal.App("hockey")

@app.function(image=image)
def notebook_image():
    pass
