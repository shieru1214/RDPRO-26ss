from recipe.augment import resolve_augmentation
from recipe.layer import build_recipe
from recipe.tables import derive_recommended_epochs


def _classification_input(**overrides):
    base = {
        "task_type": "classification",
        "data_size": "medium",
        "priority": "balanced",
        "constraints": {},
    }
    base.update(overrides)
    return base


def test_legacy_epoch_table_values_are_unchanged():
    assert derive_recommended_epochs("small", "head_only", True) == 25
    assert derive_recommended_epochs("small", "full", True) == 40
    assert derive_recommended_epochs("small", "full", False) == 50
    assert derive_recommended_epochs("medium", "head_only", True) == 12
    assert derive_recommended_epochs("medium", "full", True) == 20
    assert derive_recommended_epochs("medium", "full", False) == 30
    assert derive_recommended_epochs("large", "head_only", True) == 8
    assert derive_recommended_epochs("large", "full", True) == 15
    assert derive_recommended_epochs("large", "full", False) == 20


def test_dinov2_high_res_fine_grained_snaps_to_patch_divisor():
    recipe, provenance = build_recipe(
        {
            "task_type": "classification",
            "backbone": "dinov2",
            "checkpoint": "dinov2_base",
            "pretrained_hf_id": "facebook/dinov2-base",
            "finetune_strategy": "full",
            "use_pretrained": True,
        },
        _classification_input(constraints={"fine_grained": True}),
        {"checkpoint": {"id": "dinov2_base"}},
        data_stats={"resolution_tier": "high", "color_mode": "rgb"},
    )

    assert recipe["image_size"] == 392
    assert "snapped /14" in provenance["image_size"]


def test_learning_rate_table_hits_cnn_and_transformer_finetune():
    cnn_recipe, _ = build_recipe(
        {
            "task_type": "classification",
            "backbone": "efficientnet",
            "checkpoint": "efficientnet_b0_imagenet",
            "pretrained_hf_id": "google/efficientnet-b0",
            "finetune_strategy": "full",
            "use_pretrained": True,
        },
        _classification_input(),
        {"checkpoint": {"id": "efficientnet_b0_imagenet"}},
        data_stats={"resolution_tier": "medium", "color_mode": "rgb"},
    )
    transformer_recipe, _ = build_recipe(
        {
            "task_type": "classification",
            "backbone": "vit",
            "checkpoint": "vit_base_in21k",
            "pretrained_hf_id": "google/vit-base-patch16-224-in21k",
            "finetune_strategy": "full",
            "use_pretrained": True,
        },
        _classification_input(),
        {"checkpoint": {"id": "vit_base_in21k"}},
        data_stats={"resolution_tier": "medium", "color_mode": "rgb"},
    )

    assert cnn_recipe["learning_rate"] == 1.0e-4
    assert transformer_recipe["learning_rate"] == 3.0e-5


def test_head_only_never_gets_heavy_augmentation():
    augmentation, provenance = resolve_augmentation(
        data_size="small",
        finetune_strategy="head_only",
        constraints={"few_shot": True},
        data_stats={"color_mode": "rgb"},
    )

    assert augmentation["tier"] != "heavy"
    assert "head_only cap" in provenance


def test_grayscale_disables_color_augmentation():
    augmentation, provenance = resolve_augmentation(
        data_size="medium",
        finetune_strategy="full",
        constraints={},
        data_stats={"color_mode": "grayscale"},
    )

    assert augmentation["invariance"]["color"] is False
    assert "grayscale veto" in provenance


def test_fine_grained_keeps_crop_scale_floor():
    augmentation, provenance = resolve_augmentation(
        data_size="small",
        finetune_strategy="full",
        constraints={"fine_grained": True},
        data_stats={"color_mode": "rgb"},
    )

    assert augmentation["invariance"]["crop_scale_min"] >= 0.5
    assert "fine_grained crop floor" in provenance


def test_missing_data_stats_still_returns_recipe_with_provenance():
    recipe, provenance = build_recipe(
        {
            "task_type": "classification",
            "backbone": "efficientnet",
            "checkpoint": "efficientnet_b0_imagenet",
            "pretrained_hf_id": "google/efficientnet-b0",
            "finetune_strategy": "full",
            "use_pretrained": True,
        },
        _classification_input(),
        {"checkpoint": {"id": "efficientnet_b0_imagenet"}},
    )

    assert recipe
    assert recipe["image_size"] == 224
    assert "signal_missing: resolution_tier" in provenance["image_size"]
    assert "signal_missing: color_mode" in provenance["augmentation"]


def test_non_classification_tasks_return_empty_recipe():
    recipe, provenance = build_recipe(
        {"task_type": "object_detection", "backbone": "yolov8"},
        {"task_type": "object_detection", "data_size": "medium", "constraints": {}},
        {},
        data_stats={"resolution_tier": "medium", "color_mode": "rgb"},
    )

    assert recipe == {}
    assert provenance["status"] == "unsupported_task:object_detection"
