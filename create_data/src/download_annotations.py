"""Step 0: fetch only the annotation files the pipeline needs (images are fetched in step 2, after B1).

Every split listed in config.SPLITS is fetched up front (train ones are only *used* when val cannot fill the
target, but having them makes B1 a single pass). Large zips are read member by member over HTTP Range.
"""
import zipfile

from config import RAW, SPLITS, URLS
from utils import RemoteZip, download

VAW_FILES = {"val": ["val.json"], "test": ["test.json"], "train": ["train_part1.json", "train_part2.json"]}


def main():
    print("COCO annotations")
    coco = RemoteZip(URLS["coco_ann_zip"])
    for split in SPLITS["COCO"]:
        coco.extract(f"annotations/instances_{split}.json", RAW / f"coco/instances_{split}.json")

    # Visual Genome image metadata: url, size, coco_id (for dedup and VG image download)
    print("VG image_data")
    z = download(URLS["vg_image_data_zip"], RAW / "vg/image_data.json.zip")
    if not (RAW / "vg/image_data.json").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW / "vg")

    print("VAW")
    for name in ["attribute_types.json", "attribute_parent_types.json"]:
        download(URLS["vaw_base"] + name, RAW / "vaw" / name)
    for split in SPLITS["VAW"]:
        for name in VAW_FILES[split]:
            download(URLS["vaw_base"] + name, RAW / "vaw" / name)

    # GQA scene graphs: object inventory for every VG image (VAW uniqueness, object counts, protected boxes)
    print("GQA scene graphs")
    z = download(URLS["gqa_scene_graphs_zip"], RAW / "gqa/sceneGraphs.zip")
    with zipfile.ZipFile(z) as zf:
        for member in ["train_sceneGraphs.json", "val_sceneGraphs.json"]:
            if not (RAW / "gqa" / member).exists():
                zf.extract(member, RAW / "gqa")
    print("GQA balanced questions")
    q = RemoteZip(URLS["gqa_questions_zip"])
    for split in SPLITS["GQA"]:
        q.extract(f"{split}_balanced_questions.json", RAW / f"gqa/{split}_balanced_questions.json")

    # CLEVR: scenes + questions (reference only), not the 18GB image archive
    print("CLEVR scenes/questions")
    rz = RemoteZip(URLS["clevr_zip"])
    for split in SPLITS["CLEVR"]:
        rz.extract(f"CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json", RAW / f"clevr/CLEVR_{split}_scenes.json")
        rz.extract(f"CLEVR_v1.0/questions/CLEVR_{split}_questions.json", RAW / f"clevr/CLEVR_{split}_questions.json")
    print("done")


if __name__ == "__main__":
    main()
