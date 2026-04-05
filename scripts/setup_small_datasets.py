from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import requests
from torchvision.datasets import DTD, EuroSAT, Flowers102, OxfordIIITPet


GDRIVE_FILES = {
    "imagenet_classnames": (
        "1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF",
        "classnames.txt",
    ),
    "caltech_split": (
        "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN",
        "split_zhou_Caltech101.json",
    ),
    "oxford_pets_split": (
        "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs",
        "split_zhou_OxfordPets.json",
    ),
    "stanford_cars_split": (
        "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT",
        "split_zhou_StanfordCars.json",
    ),
    "flowers_cat_to_name": (
        "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0",
        "cat_to_name.json",
    ),
    "flowers_split": (
        "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT",
        "split_zhou_OxfordFlowers.json",
    ),
    "dtd_split": (
        "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
        "split_zhou_DescribableTextures.json",
    ),
    "eurosat_split": (
        "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o",
        "split_zhou_EuroSAT.json",
    ),
}


DIRECT_DOWNLOADS = {
    "imagenetv2": (
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz?download=true",
        "imagenetv2-matched-frequency.tar.gz",
    ),
    "imagenet_a": (
        "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
        "imagenet-a.tar",
    ),
}


STANFORD_CARS_PARQUETS = {
    "cars_train": (
        "Multimodal-Fatima/StanfordCars_train",
        [
            "data/train-00000-of-00003-0a0d1552d15aa359.parquet",
            "data/train-00001-of-00003-2e8b130aa9a8d532.parquet",
            "data/train-00002-of-00003-97e1eef4dc00b2c1.parquet",
        ],
        8144,
    ),
    "cars_test": (
        "Multimodal-Fatima/StanfordCars_test",
        [
            "data/test-00000-of-00003-18db3ba1d2223f87.parquet",
            "data/test-00001-of-00003-6e36b2afc7918744.parquet",
            "data/test-00002-of-00003-0667c9f1c47fb464.parquet",
        ],
        8041,
    ),
}


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_contents(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    ensure_dir(dst)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if not overwrite:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))


def download_gdrive(file_id: str, output: Path) -> None:
    if output.exists():
        log(f"Skip existing file: {output}")
        return

    ensure_dir(output.parent)
    url = f"https://drive.google.com/uc?id={file_id}"
    log(f"Downloading Google Drive file -> {output}")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to download Google Drive file to {output}")


def download_stream(url: str, output: Path, chunk_size: int = 1024 * 1024) -> None:
    if output.exists():
        log(f"Skip existing archive: {output}")
        return

    ensure_dir(output.parent)
    log(f"Downloading {url} -> {output}")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def extract_tar(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    log(f"Extracting {archive} -> {dest}")
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(dest)


def extract_zip(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    log(f"Extracting {archive} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def run_torchvision_downloads(root: Path) -> None:
    log("Downloading torchvision-backed datasets")
    if not (root / "oxford_pets" / "images").exists():
        OxfordIIITPet(root=str(root), split="trainval", download=True)
    else:
        log("Skip OxfordPets torchvision download")

    if not (root / "oxford_flowers" / "jpg").exists():
        Flowers102(root=str(root), split="train", download=True)
    else:
        log("Skip OxfordFlowers torchvision download")

    if not (root / "dtd" / "images").exists():
        DTD(root=str(root), split="train", download=True)
    else:
        log("Skip DTD torchvision download")

    if not (root / "eurosat" / "2750").exists():
        EuroSAT(root=str(root), download=True)
    else:
        log("Skip EuroSAT torchvision download")


def download_caltech(root: Path) -> None:
    target_dir = root / "caltech-101"
    if (target_dir / "101_ObjectCategories").exists():
        log("Caltech101 already present")
        return

    archives_dir = ensure_dir(root / "_archives")
    archive = archives_dir / "caltech-101.zip"
    download_stream(
        "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
        archive,
    )
    extract_zip(archive, target_dir)
    nested_dir = target_dir / "caltech-101"
    nested_archive = nested_dir / "101_ObjectCategories.tar.gz"
    if nested_archive.exists():
        extract_tar(nested_archive, target_dir)


def normalize_caltech(root: Path) -> None:
    src = root / "caltech101"
    dst = ensure_dir(root / "caltech-101")
    if (dst / "101_ObjectCategories").exists():
        log("Caltech101 already normalized")
        return
    if not src.exists():
        raise FileNotFoundError(f"Expected Caltech download folder at {src}")
    move_contents(src, dst)


def normalize_oxford_pets(root: Path) -> None:
    src = root / "oxford-iiit-pet"
    dst = ensure_dir(root / "oxford_pets")
    if (dst / "images").exists() and (dst / "annotations").exists():
        log("OxfordPets already normalized")
        return
    move_contents(src, dst)


def normalize_oxford_flowers(root: Path) -> None:
    src = root / "flowers-102"
    dst = ensure_dir(root / "oxford_flowers")
    if (dst / "jpg").exists() and (dst / "imagelabels.mat").exists():
        log("OxfordFlowers already normalized")
        return
    move_contents(src, dst)


def normalize_dtd(root: Path) -> None:
    base = ensure_dir(root / "dtd")
    nested = base / "dtd"
    if (base / "images").exists() and (base / "labels").exists():
        log("DTD already normalized")
        return
    if not nested.exists():
        raise FileNotFoundError(f"Expected extracted DTD folder at {nested}")
    move_contents(nested, base)
    shutil.rmtree(nested)


def cleanup_artifacts(root: Path) -> None:
    log("Cleaning nested wrappers and redundant archives")

    caltech_dir = root / "caltech-101"
    nested_caltech = caltech_dir / "caltech-101"
    if nested_caltech.exists():
        shutil.rmtree(nested_caltech)
    macosx_dir = caltech_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    removable_patterns = {
        root / "oxford_pets": ["*.tar.gz"],
        root / "oxford_flowers": ["*.tgz"],
        root / "dtd": ["*.tar.gz"],
        root / "eurosat": ["*.zip"],
    }

    for folder, patterns in removable_patterns.items():
        if not folder.exists():
            continue
        for pattern in patterns:
            for file_path in folder.glob(pattern):
                file_path.unlink(missing_ok=True)

    source_cleanup = [
        (root / "flowers-102", root / "oxford_flowers" / "jpg"),
        (root / "oxford-iiit-pet", root / "oxford_pets" / "images"),
    ]
    for source_dir, normalized_probe in source_cleanup:
        if source_dir.exists() and normalized_probe.exists():
            shutil.rmtree(source_dir)

    empty_dirs = [root / "caltech101"]
    for empty_dir in empty_dirs:
        if empty_dir.exists() and not any(empty_dir.iterdir()):
            empty_dir.rmdir()


def download_imagenet_variants(root: Path) -> None:
    archives_dir = ensure_dir(root / "_archives")

    imagenetv2_root = ensure_dir(root / "imagenetv2")
    if not (imagenetv2_root / "imagenetv2-matched-frequency-format-val").exists():
        url, filename = DIRECT_DOWNLOADS["imagenetv2"]
        archive = archives_dir / filename
        download_stream(url, archive)
        extract_tar(archive, imagenetv2_root)
    else:
        log("ImageNetV2 already present")

    imagenet_a_root = ensure_dir(root / "imagenet-adversarial")
    if not (imagenet_a_root / "imagenet-a").exists():
        url, filename = DIRECT_DOWNLOADS["imagenet_a"]
        archive = archives_dir / filename
        download_stream(url, archive)
        extract_tar(archive, imagenet_a_root)
    else:
        log("ImageNet-A already present")


def setup_metadata(root: Path) -> None:
    download_gdrive(*_metadata_args("caltech_split", root / "caltech-101"))
    download_gdrive(*_metadata_args("oxford_pets_split", root / "oxford_pets"))
    download_gdrive(*_metadata_args("flowers_cat_to_name", root / "oxford_flowers"))
    download_gdrive(*_metadata_args("flowers_split", root / "oxford_flowers"))
    download_gdrive(*_metadata_args("dtd_split", root / "dtd"))
    download_gdrive(*_metadata_args("eurosat_split", root / "eurosat"))

    imagenet_classnames_id, imagenet_classnames_name = GDRIVE_FILES["imagenet_classnames"]
    classnames_cache = root / "_metadata_cache" / imagenet_classnames_name
    download_gdrive(imagenet_classnames_id, classnames_cache)

    for dataset_dir in [root / "imagenetv2", root / "imagenet-adversarial"]:
        target = dataset_dir / "classnames.txt"
        if target.exists():
            log(f"Skip existing file: {target}")
            continue
        ensure_dir(dataset_dir)
        shutil.copy2(classnames_cache, target)
        log(f"Copied classnames.txt -> {target}")


def _metadata_args(key: str, folder: Path) -> tuple[str, Path]:
    file_id, filename = GDRIVE_FILES[key]
    return file_id, folder / filename


def _extract_stanford_cars_shard(parquet_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(columns=["image"], batch_size=256):
        for row in batch.to_pylist():
            image_info = row["image"]
            out_path = out_dir / image_info["path"]
            if out_path.exists():
                continue
            out_path.write_bytes(image_info["bytes"])


def setup_stanford_cars(root: Path) -> None:
    log("Setting up StanfordCars from Hugging Face parquet shards")
    target_dir = ensure_dir(root / "stanford_cars")
    download_gdrive(*_metadata_args("stanford_cars_split", target_dir))
    hf_cache = ensure_dir(root / "_hf_cache_stanford")

    for split_dir_name, (repo_id, parquet_files, expected_count) in STANFORD_CARS_PARQUETS.items():
        out_dir = ensure_dir(target_dir / split_dir_name)
        existing_count = sum(1 for _ in out_dir.glob("*.jpg"))
        if existing_count >= expected_count:
            log(f"StanfordCars {split_dir_name} already present ({existing_count} files)")
            continue

        log(f"Downloading StanfordCars shards for {split_dir_name}")
        for filename in parquet_files:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(hf_cache),
                )
            )
            _extract_stanford_cars_shard(local_path, out_dir)

        final_count = sum(1 for _ in out_dir.glob("*.jpg"))
        if final_count != expected_count:
            raise RuntimeError(
                f"StanfordCars {split_dir_name} expected {expected_count} files, found {final_count}"
            )

    if hf_cache.exists():
        shutil.rmtree(hf_cache)


def summarize(root: Path, dataset_names: Iterable[str]) -> None:
    log("Final dataset layout")
    for name in dataset_names:
        path = root / name
        print(f"- {name}: {path} {'[OK]' if path.exists() else '[MISSING]'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--skip-stanford-cars",
        action="store_true",
        help="Skip StanfordCars setup",
    )
    args = parser.parse_args()

    root = ensure_dir(args.root.resolve())
    log(f"Dataset root: {root}")

    download_caltech(root)
    run_torchvision_downloads(root)
    normalize_oxford_pets(root)
    normalize_oxford_flowers(root)
    normalize_dtd(root)
    download_imagenet_variants(root)
    setup_metadata(root)
    cleanup_artifacts(root)

    if not args.skip_stanford_cars:
        setup_stanford_cars(root)

    summarize(
        root,
        [
            "caltech-101",
            "oxford_pets",
            "oxford_flowers",
            "dtd",
            "eurosat",
            "imagenetv2",
            "imagenet-adversarial",
            "stanford_cars",
        ],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        raise
