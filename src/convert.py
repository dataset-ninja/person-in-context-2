import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_path = "/home/alex/DATASETS/TODO/PIC/PIC_2.0/image/train"
    val_path = "/home/alex/DATASETS/TODO/PIC/PIC_2.0/image/val"
    test_path = "/home/alex/DATASETS/TODO/PIC/PIC_2.0/image/test"
    images_folder = "image"
    anns_folder = "semantic"
    batch_size = 30
    labels_path = "/home/alex/DATASETS/TODO/PIC/PIC_2.0/categories_list/label_categories.json"
    # relation_categories_path = (
    #     "/home/alex/DATASETS/TODO/PIC/PIC_2.0/categories_list/relation_categories.json"
    # )
    # relations_train_path = "/home/alex/DATASETS/TODO/PIC/PIC_2.0/relations_train.json"
    # relations_val_path = "/home/alex/DATASETS/TODO/PIC/PIC_2.0/relations_val.json"

    ds_name_to_pathes = {"train": train_path, "val": val_path, "test": test_path}

    def create_ann(image_path):
        labels = []
        tags = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        if get_file_name(image_path)[:6] == "indoor":
            tag = sly.Tag(indoor_meta)
            tags.append(tag)
        elif get_file_name(image_path)[:6] == "outdoo":
            tag = sly.Tag(outdoor_meta)
            tags.append(tag)

        mask_path = image_path.replace(images_folder, anns_folder).replace(".jpg", ".png")

        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            unique_pixels = np.unique(mask_np)[1:]
            for curr_pixel in unique_pixels:
                obj_class = pixel_to_class[curr_pixel]
                mask = mask_np == curr_pixel
                ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                for i in range(1, ret):
                    obj_mask = curr_mask == i
                    curr_bitmap = sly.Bitmap(obj_mask)
                    if curr_bitmap.area > 25:
                        curr_label = sly.Label(curr_bitmap, obj_class)
                        labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    pixel_to_class = {}
    labels = load_json_file(labels_path)
    for curr_label in labels:
        pixel_to_class[curr_label["id"]] = sly.ObjClass(curr_label["name"], sly.Bitmap)

    indoor_meta = sly.TagMeta("indoor", sly.TagValueType.NONE)
    outdoor_meta = sly.TagMeta("outdoor", sly.TagValueType.NONE)

    meta = sly.ProjectMeta(
        obj_classes=list(pixel_to_class.values()), tag_metas=[indoor_meta, outdoor_meta]
    )
    api.project.update_meta(project.id, meta.to_json())

    # idx_to_name = {}
    # relation_categories = load_json_file(relation_categories_path)
    # for rel_cat in relation_categories:
    #     idx_to_name[rel_cat["id"]] = rel_cat["name"]

    # relations_train = load_json_file(relations_train_path)

    for ds_name, images_path in ds_name_to_pathes.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(images_path, im_name) for im_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
