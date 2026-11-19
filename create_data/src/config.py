"""Shared paths and parameters for the dataset pipeline (v2, 2026-11-19).

Steps: 0 annotations -> 1 B1 candidates -> 2 image selection (+ VLM check) -> [human review] -> 3 B2 ->
4 baseline -> 5 B3 negative -> 6 affirmative/exceptive forms -> 7 export -> 8 visual -> (cosyvoice env) TTS.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
RAW = DATA / "raw"            # downloaded annotations (untouched)
IMAGES = DATA / "images"      # selected images, one folder per source
REJECTED = DATA / "images_rejected"  # downloaded but rejected by a pixel / VLM check (kept for the review page)
B1_DIR = DATA / "b1"          # candidates, selected units, selection report, review page
B2_DIR = DATA / "b2"          # verified atomic components (+ baseline answers)
OUT_DIR = DATA / "out"        # final samples
VIS = DATA / "visual"         # visual-channel images

SEED = 42

# ---------------------------------------------------------------- sources and splits
# Splits are used in this order; a later split only fills what the earlier ones could not supply.
# (COCO / GQA / CLEVR test sets have no public annotation; VAW test does.)
SPLITS = {
    "COCO": ["val2017", "train2017"],
    "VAW": ["val", "test", "train"],
    "GQA": ["val", "train"],
    "CLEVR": ["val"],
}
# Images to select per source (after every image-level check, before the baseline step).
# v2 targets include headroom: the baseline step and human review drop more later (~2.5k final images).
# GQA relation is the scarce branch: take every image that passes (~450-600 expected).
TARGET_IMAGES = {"COCO": 950, "VAW": 950, "GQA": 10 ** 6, "CLEVR": 600}

# ---------------------------------------------------------------- B1: annotation-level filters
MIN_COMPONENTS = 2
MAX_COMPONENTS = 5
MAX_COUNT = 5                 # count component only for categories with <= 5 instances
MIN_AREA_RATIO = 0.03         # every component's object covers >= 3% of the image
MIN_COUNT_INSTANCE_RATIO = 0.01   # count questions: every counted instance >= 1% (tiny ones make counts unclear)
BORDER_PX = 2                 # the component's (largest) instance must not touch the image border
MIN_SHORT_SIDE = 300          # px (VG images are mostly 500x333 / 500x375)
MAX_OBJECTS = {"COCO": 10, "VG": 10, "CLEVR": 6}   # COCO instances / VG "things" (b1.NOT_CLUTTER not counted) / CLEVR
MIN_RELATION_OBJECT_RATIO = 0.01  # relation: the related object (cup, phone...) only needs 1%; the subject needs 3%
COCO_EXCLUDED_LICENSES = {3, 6}    # NoDerivs (the visual channel writes on the image = a derivative work)
N_DISTRACTORS = 2             # absent same-supercategory categories in a closed-set existence question

# ---------------------------------------------------------------- step 2: pixel-level filters
MIN_SHARPNESS = 300           # variance of the Laplacian, grey image resized to 512 px wide (not CLEVR)
BRIGHTNESS_RANGE = (40, 215)  # mean grey level
MAX_CLIPPED = 0.35            # share of pixels <= 5 or >= 250
MIN_COLORFULNESS = 4.0        # mean |R-G| + |G-B| (rejects black-and-white photos)
DHASH_MAX_DIST = 6            # near-duplicate: 64-bit dHash Hamming distance <= this to an accepted image
OTHER_WEIGHT = 0.15           # protected-box weight of objects outside the question (text placement)
MAX_SUBJECT_COVER = 0.05      # text box may cover at most 5% of the question's object boxes (boxes are loose rectangles)
VERIFY_MODEL = "Qwen/Qwen3-VL-8B-Instruct"   # yes/no check of every component, distractor and big text

# ---------------------------------------------------------------- B2
HUMAN_VERIFY_RATE = 0.10
ERROR_RATE_THRESHOLD = 0.05
B2_MODEL = "Qwen/Qwen3-8B"           # vLLM fails on this machine (flashinfer vs py3.10), so transformers fp16
B2_PROMPT_VERSION = "v2"

# ---------------------------------------------------------------- baseline (reference models) and B3
BASELINE_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]
BASELINE_MIN_REQUIRED = 0.5   # each reference model must state >= half of the required components
FULL_TEMPLATE_SUBSET_RATE = 0.10   # share of units expanded to all 7 negative templates

URLS = {
    "coco_ann_zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "coco_img": "http://images.cocodataset.org/{split}/{id:012d}.jpg",
    "vg_image_data_zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    "vaw_base": "https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/",
    "gqa_scene_graphs_zip": "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
    "gqa_questions_zip": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
    "clevr_zip": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
}
