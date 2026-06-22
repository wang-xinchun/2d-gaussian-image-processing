#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "papers" / "catalog"

MANIFEST_SPECS = [
    ("batch_01", ROOT / "papers" / "top_venues" / "manifest.tsv"),
    ("batch_02", ROOT / "papers" / "top_venues" / "batch_02" / "manifest.tsv"),
    ("batch_03", ROOT / "papers" / "top_venues" / "batch_03" / "manifest.tsv"),
    ("batch_04", ROOT / "papers" / "top_venues" / "batch_04" / "manifest.tsv"),
]

TOPICS = [
    (
        "general_restoration_backbone",
        "通用图像复原骨干",
        "面向去噪、去模糊、去雨、增强等多类任务的通用复原主干与高分辨率友好架构。",
    ),
    (
        "all_in_one_complex_degradation",
        "全任务/复杂退化复原",
        "面向未知退化、复杂退化、统一复原或 all-in-one 复原的问题设定。",
    ),
    (
        "sr_backbone_arbitrary_scale",
        "超分辨率主干与任意倍率",
        "以超分网络设计、SR Transformer、任意倍率 SR 为主的结构类论文。",
    ),
    (
        "real_world_sr_degradation_modeling",
        "真实世界超分与退化建模",
        "以真实图像超分、真实退化建模、数据生成与 benchmark 为主的论文。",
    ),
    (
        "diffusion_generative_sr",
        "扩散/生成式超分与图像复原",
        "以 diffusion、autoregressive、VAE、multimodal 等生成式先验为主的 SR/复原论文。",
    ),
]

TOPIC_LABELS = {topic_id: label for topic_id, label, _ in TOPICS}
TOPIC_DESCRIPTIONS = {topic_id: description for topic_id, _, description in TOPICS}
TOPIC_ORDER = {topic_id: index for index, (topic_id, _, _) in enumerate(TOPICS)}

VENUE_ORDER = {
    "TPAMI": 0,
    "CVPR": 1,
    "ICCV": 2,
    "ECCV": 3,
    "NeurIPS": 4,
}

CLASSIFICATION = {
    "papers/top_venues/conference/2019_CVPR_DualResidual_ImageRestoration.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["cnn", "paired-operations"],
    },
    "papers/top_venues/conference/2021_CVPR_MPRNet_MultiStageProgressiveImageRestoration.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["multi-stage", "progressive"],
    },
    "papers/top_venues/conference/2022_CVPR_Uformer_ImageRestoration.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["transformer", "u-shaped"],
    },
    "papers/top_venues/conference/2022_CVPR_Restormer_HighResolutionImageRestoration.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["transformer", "high-resolution-friendly"],
    },
    "papers/top_venues/conference/2022_CVPR_AirNet_AllInOneImageRestoration.pdf": {
        "topic": "all_in_one_complex_degradation",
        "tags": ["all-in-one", "unknown-corruption"],
    },
    "papers/top_venues/conference/2022_ECCV_NAFNet_SimpleBaselinesImageRestoration.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["efficient", "simple-baseline"],
    },
    "papers/top_venues/conference/2023_CVPR_HAT_ActivatingMorePixels_SRTransformer.pdf": {
        "topic": "sr_backbone_arbitrary_scale",
        "tags": ["transformer", "super-resolution"],
    },
    "papers/top_venues/conference/2023_ICCV_DAT_DualAggregationTransformer_SR.pdf": {
        "topic": "sr_backbone_arbitrary_scale",
        "tags": ["transformer", "super-resolution"],
    },
    "papers/top_venues/conference/2023_NeurIPS_PromptIR_AllInOneBlindImageRestoration.pdf": {
        "topic": "all_in_one_complex_degradation",
        "tags": ["all-in-one", "blind-restoration", "prompt-learning"],
    },
    "papers/top_venues/journal/2022_TPAMI_SR3_ImageSuperResolutionViaIterativeRefinement.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "super-resolution"],
    },
    "papers/top_venues/journal/2023_TPAMI_MIRNetv2_FastImageRestorationAndEnhancement.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["restoration", "enhancement"],
    },
    "papers/top_venues/conference/2025_ICCV_TEAFormer_AdaptiveTranslationEquivariance.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["transformer", "high-resolution-friendly"],
    },
    "papers/top_venues/batch_02/conference/2021_CVPR_IPT_PreTrainedImageProcessingTransformer.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["pretraining", "transformer"],
    },
    "papers/top_venues/batch_02/conference/2023_CVPR_EQSR_ArbitraryScaleImageSuperResolution.pdf": {
        "topic": "sr_backbone_arbitrary_scale",
        "tags": ["arbitrary-scale", "super-resolution"],
    },
    "papers/top_venues/batch_02/conference/2023_CVPR_CiaoSR_ArbitraryScaleImageSuperResolution.pdf": {
        "topic": "sr_backbone_arbitrary_scale",
        "tags": ["arbitrary-scale", "implicit-representation"],
    },
    "papers/top_venues/batch_02/conference/2024_CVPR_COZ_ContinuousOpticalZooming.pdf": {
        "topic": "real_world_sr_degradation_modeling",
        "tags": ["benchmark", "arbitrary-scale", "real-world"],
    },
    "papers/top_venues/batch_02/conference/2024_CVPR_IPG_ImageProcessingGNN_SuperResolution.pdf": {
        "topic": "sr_backbone_arbitrary_scale",
        "tags": ["gnn", "super-resolution"],
    },
    "papers/top_venues/batch_02/conference/2024_CVPR_SeeSR_RealWorldImageSuperResolution.pdf": {
        "topic": "real_world_sr_degradation_modeling",
        "tags": ["real-world", "semantic-prior"],
    },
    "papers/top_venues/batch_02/conference/2024_CVPR_SinSR_DiffusionBasedSingleStepSR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "single-step"],
    },
    "papers/top_venues/batch_02/conference/2025_ICCV_PURE_RealWorldImageSuperResolution.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["autoregressive", "multimodal", "real-world"],
    },
    "papers/top_venues/batch_02/conference/2025_ICCV_TVT_FinestructurePreservedRealWorldSR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["vae", "real-world"],
    },
    "papers/top_venues/batch_02/conference/2025_ICCV_UniRes_UniversalImageRestoration.pdf": {
        "topic": "all_in_one_complex_degradation",
        "tags": ["universal-restoration", "complex-degradations"],
    },
    "papers/top_venues/batch_02/journal/2025_TPAMI_HAT_HybridAttentionTransformerImageRestoration.pdf": {
        "topic": "sr_backbone_arbitrary_scale",
        "tags": ["transformer", "super-resolution"],
    },
    "papers/top_venues/batch_03/conference/2019_ICCV_RealSR_NewBenchmarkAndModel.pdf": {
        "topic": "real_world_sr_degradation_modeling",
        "tags": ["benchmark", "real-world"],
    },
    "papers/top_venues/batch_03/conference/2019_ICCV_KMSR_RealLowResolutionImages.pdf": {
        "topic": "real_world_sr_degradation_modeling",
        "tags": ["kernel-modeling", "real-world"],
    },
    "papers/top_venues/batch_03/conference/2021_ICCV_BSRGAN_PracticalDegradationModel_BlindSR.pdf": {
        "topic": "real_world_sr_degradation_modeling",
        "tags": ["degradation-modeling", "blind-sr"],
    },
    "papers/top_venues/batch_03/conference/2021_ICCV_DARealSR_DomainAdaptationPerspective.pdf": {
        "topic": "real_world_sr_degradation_modeling",
        "tags": ["domain-adaptation", "real-world"],
    },
    "papers/top_venues/batch_03/conference/2023_NeurIPS_ResShift_DiffusionModelForImageSR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "super-resolution"],
    },
    "papers/top_venues/batch_03/conference/2024_CVPR_CDFormer_BlindImageSuperResolution.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "blind-sr", "degradation-prediction"],
    },
    "papers/top_venues/batch_03/conference/2025_CVPR_UPSRDM_UncertaintyGuidedPerturbation_SRDiffusion.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "uncertainty"],
    },
    "papers/top_venues/batch_03/conference/2025_CVPR_ASDI_ArbitraryStepsImageSR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "diffusion-inversion"],
    },
    "papers/top_venues/batch_03/conference/2025_CVPR_TSDSR_OneStepDiffusion_RealWorldISR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "one-step", "real-world"],
    },
    "papers/top_venues/batch_03/conference/2025_CVPR_AdcSR_AdversarialDiffusionCompression_RealWorldISR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "compression", "real-world"],
    },
    "papers/top_venues/batch_03/conference/2025_CVPR_MMSR_PowerOfContext_MultimodalitySR.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["multimodal", "super-resolution"],
    },
    "papers/top_venues/batch_03/conference/2025_ICCV_DiT4SR_RealWorldImageSuperResolution.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion-transformer", "real-world"],
    },
    "papers/top_venues/batch_03/conference/2025_ICCV_LDRPS_ZeroShotUnifiedImageRestoration.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["latent-diffusion", "zero-shot", "unified-restoration"],
    },
    "papers/top_venues/batch_04/conference/2022_CVPR_MAXIM_MultiAxisMLP_ImageProcessing.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["mlp", "multi-task", "high-resolution-friendly"],
    },
    "papers/top_venues/batch_04/conference/2022_CVPR_TransWeather_AdverseWeatherRestoration.pdf": {
        "topic": "all_in_one_complex_degradation",
        "tags": ["adverse-weather", "transformer", "all-in-one"],
    },
    "papers/top_venues/batch_04/conference/2022_ECCV_TLC_GlobalInformationAggregation_ImageRestoration.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["test-time-local-converter", "patch-vs-image", "high-resolution-friendly"],
    },
    "papers/top_venues/batch_04/conference/2022_ECCV_Stripformer_FastImageDeblurring.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["transformer", "deblurring", "efficient"],
    },
    "papers/top_venues/batch_04/conference/2021_ICCV_MIMO_UNet_RethinkingCoarseToFine_Deblurring.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["deblurring", "multi-scale", "efficient"],
    },
    "papers/top_venues/batch_04/conference/2023_CVPR_FFTformer_FrequencyDomainTransformers_Deblurring.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["frequency-domain", "transformer", "deblurring"],
    },
    "papers/top_venues/batch_04/conference/2023_CVPR_RIDCP_RealImageDehazing_CodebookPriors.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["dehazing", "real-image", "codebook-prior"],
    },
    "papers/top_venues/batch_04/conference/2024_CVPR_PTG_RM_BoostingImageRestoration_PretrainedPriors.pdf": {
        "topic": "general_restoration_backbone",
        "tags": ["pretrained-prior", "lightweight-refinement", "multi-task"],
    },
    "papers/top_venues/batch_04/conference/2024_CVPR_DTPM_LearningDiffusionTexturePriors_ImageRestoration.pdf": {
        "topic": "diffusion_generative_sr",
        "tags": ["diffusion", "texture-prior", "image-restoration"],
    },
}


def load_manifest(batch_name: str, manifest_path: Path) -> list[dict[str, str]]:
    papers: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(
            (line for line in handle if not line.startswith("#")),
            delimiter="\t",
            fieldnames=[
                "category",
                "venue",
                "year",
                "short_name",
                "title",
                "pdf_url",
                "official_source",
                "note",
                "local_rel_path",
            ],
        )
        for row in reader:
            if not row["category"]:
                continue
            local_rel_path = row["local_rel_path"]
            if local_rel_path not in CLASSIFICATION:
                raise KeyError(f"Missing classification for {local_rel_path}")

            pdf_path = ROOT / local_rel_path
            if not pdf_path.exists():
                raise FileNotFoundError(f"Missing PDF file: {pdf_path}")

            topic_id = CLASSIFICATION[local_rel_path]["topic"]
            tags = CLASSIFICATION[local_rel_path]["tags"]
            papers.append(
                {
                    **row,
                    "batch": batch_name,
                    "topic_id": topic_id,
                    "topic_label": TOPIC_LABELS[topic_id],
                    "tags": ", ".join(tags),
                    "pdf_rel_from_catalog": str(pdf_path.relative_to(CATALOG_DIR.parent)),
                    "source_link": row["official_source"],
                }
            )
    return papers


def sort_key(paper: dict[str, str]) -> tuple[int, int, str]:
    return (
        int(paper["year"]),
        VENUE_ORDER.get(paper["venue"], 99),
        paper["title"],
    )


def build_paper_index(papers: list[dict[str, str]]) -> None:
    headers = [
        "paper_id",
        "batch",
        "category",
        "venue",
        "year",
        "short_name",
        "title",
        "primary_topic",
        "tags",
        "local_rel_path",
        "official_source",
    ]
    lines = ["\t".join(headers)]
    for index, paper in enumerate(papers, start=1):
        row = [
            f"P{index:04d}",
            paper["batch"],
            paper["category"],
            paper["venue"],
            paper["year"],
            paper["short_name"],
            paper["title"],
            paper["topic_label"],
            paper["tags"],
            paper["local_rel_path"],
            paper["official_source"],
        ]
        lines.append("\t".join(row))
    (CATALOG_DIR / "paper_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_topic_summary(papers: list[dict[str, str]]) -> Counter:
    counts: Counter[str] = Counter(paper["topic_id"] for paper in papers)
    lines = ["topic_id\ttopic_label\tpaper_count"]
    for topic_id, label, _ in TOPICS:
        lines.append(f"{topic_id}\t{label}\t{counts[topic_id]}")
    (CATALOG_DIR / "topic_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def build_venue_summary(papers: list[dict[str, str]]) -> Counter:
    counts: Counter[str] = Counter(paper["venue"] for paper in papers)
    venues = sorted(counts, key=lambda venue: (VENUE_ORDER.get(venue, 99), venue))
    lines = ["venue\tpaper_count"]
    for venue in venues:
        lines.append(f"{venue}\t{counts[venue]}")
    (CATALOG_DIR / "venue_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def build_topic_markdown(papers: list[dict[str, str]], topic_counts: Counter) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for paper in papers:
        grouped[paper["topic_id"]].append(paper)

    source_range = " + ".join(batch_name for batch_name, _ in MANIFEST_SPECS)
    lines = [
        "# 按题材分类",
        "",
        f"- 论文总数：`{len(papers)}`",
        f"- 来源范围：`{source_range}`",
        "",
    ]

    for topic_id, label, description in TOPICS:
        lines.append(f"## {label} ({topic_counts[topic_id]})")
        lines.append("")
        lines.append(f"- 说明：{description}")
        lines.append("")
        for paper in sorted(grouped[topic_id], key=sort_key):
            lines.append(
                f"- [{paper['year']} {paper['venue']}] "
                f"[{paper['title']}](../{paper['local_rel_path'].split('papers/', 1)[1]}) "
                f"| `{paper['batch']}` | tags: `{paper['tags']}`"
            )
        lines.append("")

    (CATALOG_DIR / "by_topic.md").write_text("\n".join(lines), encoding="utf-8")


def build_venue_markdown(papers: list[dict[str, str]], venue_counts: Counter) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for paper in papers:
        grouped[paper["venue"]].append(paper)

    lines = [
        "# 按期刊会议分类",
        "",
        f"- 论文总数：`{len(papers)}`",
        "- venue 范围：`TPAMI / CVPR / ICCV / ECCV / NeurIPS`",
        "",
    ]

    for venue in sorted(grouped, key=lambda key: (VENUE_ORDER.get(key, 99), key)):
        lines.append(f"## {venue} ({venue_counts[venue]})")
        lines.append("")
        for paper in sorted(grouped[venue], key=sort_key):
            lines.append(
                f"- [{paper['year']}] [{paper['title']}](../{paper['local_rel_path'].split('papers/', 1)[1]}) "
                f"| `{paper['batch']}` | topic: `{paper['topic_label']}`"
            )
        lines.append("")

    (CATALOG_DIR / "by_venue.md").write_text("\n".join(lines), encoding="utf-8")


def build_readme(
    papers: list[dict[str, str]],
    topic_counts: Counter,
    venue_counts: Counter,
) -> None:
    batch_counts = Counter(paper["batch"] for paper in papers)
    source_range = " + ".join(batch_name for batch_name, _ in MANIFEST_SPECS)
    lines = [
        "# 论文总目录",
        "",
        "这里汇总了当前项目内所有已下载顶刊顶会论文批次的总目录与分类索引。",
        "",
        "## 文件说明",
        "",
        "- `paper_index.tsv`: 机器可读的总索引。",
        "- `topic_summary.tsv`: 题材统计汇总。",
        "- `venue_summary.tsv`: venue 统计汇总。",
        "- `by_topic.md`: 按题材整理的可读目录。",
        "- `by_venue.md`: 按期刊会议整理的可读目录。",
        "",
        "## 总览",
        "",
        f"- 论文总数：`{len(papers)}`",
        f"- 来源范围：`{source_range}`",
    ]

    for batch_name, _ in MANIFEST_SPECS:
        lines.append(f"- `{batch_name}`: `{batch_counts[batch_name]}`")

    lines.extend(
        [
            "",
            "## 题材体系",
            "",
        ]
    )

    for topic_id, label, description in TOPICS:
        lines.append(f"- `{label}`: `{topic_counts[topic_id]}`")
        lines.append(f"  说明：{description}")

    lines.extend(
        [
            "",
            "## Venue 统计",
            "",
        ]
    )

    for venue in sorted(venue_counts, key=lambda key: (VENUE_ORDER.get(key, 99), key)):
        lines.append(f"- `{venue}`: `{venue_counts[venue]}`")

    lines.extend(
        [
            "",
            "## 重新生成",
            "",
            "- 在项目根目录运行 `python3 scripts/build_paper_catalog.py`。",
            "",
        ]
    )

    (CATALOG_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    papers: list[dict[str, str]] = []
    for batch_name, manifest_path in MANIFEST_SPECS:
        papers.extend(load_manifest(batch_name, manifest_path))

    papers = sorted(papers, key=sort_key)
    build_paper_index(papers)
    topic_counts = build_topic_summary(papers)
    venue_counts = build_venue_summary(papers)
    build_topic_markdown(papers, topic_counts)
    build_venue_markdown(papers, venue_counts)
    build_readme(papers, topic_counts, venue_counts)


if __name__ == "__main__":
    main()
