# V1/T01 Training Template

`T01` is the first reusable training template for the `V1` architecture.

It is intentionally conservative:

- paired supervised restoration
- `UHD-LL_down4` as the first short-training dataset
- Charbonnier reconstruction loss plus a small gradient loss
- patch-based training
- validation PSNR on a limited validation subset
- independent run archive under `experiments/V1/runs/<run_id>`

Primary entrypoint:

```bash
python3 scripts/train_v1_t01.py --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml
```

Current prepared configs:

- `configs/a01_short_uhdll_down4.toml`: first low-light downsampled deployment.
- `configs/a02_short_uhdhaze_patch.toml`: first cross-task UHD dehazing short probe on `UHD_haze`.
- `configs/a_a02_short_uhdhaze_patch_iobalanced.toml`: data-loader-balanced derivative of A02.
- `configs/a03_short_uhddeblur_patch.toml`: accepted first UHD deblurring baseline on `UHD_deblur`.
- `configs/a_a03_short_uhddeblur_capacity.toml`: undeployed `A03` branch design that increases model capacity and Gaussian expression for `UHD_deblur`; it still requires smoke calibration, formal preflight, and explicit launch approval.
- `configs/a_b03_short_uhddeblur_fullval.toml`: `A_a03`-capacity `UHD_deblur` branch that changes training-time validation to full-image `crop_size=0` internal validation to better align checkpoint selection with the official protocol.
- `configs/a_c03_short_uhddeblur_fullval_vram80.toml`: high-VRAM replacement for `A_b03`, preserving full-image internal validation while raising `batch_size` to `25` to satisfy the `>=80G` training-memory floor.
- `configs/a_d03_short_uhddeblur_edge_guided_vram80.toml`: high-VRAM `UHD_deblur` branch that adds the default-off high-resolution edge-guided detail refiner plus a small Laplacian consistency loss.
- `configs/a_e03_short_uhddeblur_edge_guided_cache_safe.toml`: cache-safe redeployment of the `A_d03` edge-guided branch after post-validation CUDA fragmentation caused an OOM at step 500.
- `configs/a_f03_short_uhddeblur_edge_guided_extended_budget.toml`: `A_e03` follow-up branch that keeps the accepted edge-guided cache-safe mechanism fixed and extends the short-run step budget from `1500` to `2500` to test whether the prior run was under-trained.
- `configs/a_g03_short_uhddeblur_directional_frequency.toml`: accepted post-`A_f03` directional/frequency branch on `UHD_deblur`; it keeps the edge-guided cache-safe baseline but injects Gaussian-derived directional/frequency priors into the high-resolution detail refiner, and its accepted `final.pt` currently defines the official `UHD_deblur` baseline.
- `configs/a_h03_short_uhddeblur_tlc_consistency.toml`: prepared post-milestone `UHD_deblur` branch after `A_g03`; it keeps the accepted edge-guided + directional-frequency baseline fixed, but adds a default-off `TLC`-style train-inference consistency mechanism that trains on `768x768` context while supervising only the centered `512x512` region. It is prepared but not yet smoke-calibrated, preflighted, or launched.
- `configs/a_i03_medium_uhddeblur_tlc_consistency_vram_window.toml`: accepted and frozen `UHD_deblur` baseline branch under milestone `M02`; it keeps the accepted `A_g03` mainline plus the `A_h03` consistency mechanism, expands the training budget to `300` minutes, enlarges the effective train split by reducing `train_offset` from `100` to `40`, and uses the accepted `batch_size=7`, `train_context_size=784` setting inside the strict `85-93G` runtime window.
- `configs/a_j03_short_uhddeblur_tlc_context864_probe.toml`: completed post-`A_i03/M02` short probe that kept the accepted `A_i03` mainline and effective train split fixed, but reallocated the same `85-93G` VRAM envelope toward a larger consistency-training context target (`864`) under a short budget; it is now a negative screening result and should not be promoted directly to the next long run.
- `configs/a_k03_medium_uhddeblur_tlc_stepcap_extension.toml`: active post-`A_j03` long-budget follow-up that keeps the accepted `A_i03/M02` setting fixed, preserves `batch_size=7`, `train_context_size=784`, `train_offset=40`, and `wall_clock_minutes=300`, but raises `max_steps` from `12000` to `24000` so the branch is no longer prematurely stopped by the earlier step cap.
- `configs/a_l03_medium_uhddeblur_gaussian_decoder_guidance_design_only.toml`: design-only post-`A_k03/M03` architectural branch that keeps the accepted edge-guided + directional/frequency + consistency line fixed but adds default-off Gaussian-guided decoder modulation before final RGB reconstruction.
- `configs/a_m03_short_uhddeblur_gaussian_decoder_guidance_lowstrength.toml`: first completed short scout for the `A_l03` idea; it enables Gaussian-guided decoder modulation with conservative `strength=0.10` at `batch_size=6`, `train_context_size=784` and is used only for probe-family screening, not for baseline replacement.
- `configs/a_n03_short_uhddeblur_gaussian_decoder_guidance_default.toml`: default-strength follow-up to `A_m03`; it changes only `gaussian_decoder_guidance_strength=0.15` while keeping the same `6x784` short-probe geometry.
- `configs/a_o03_short_uhddeblur_gaussian_decoder_guidance_highfill_probe.toml`: high-fill runtime scout for the same mechanism family; it keeps default strength `0.15` but changes launch geometry to `batch_size=7`, `train_context_size=776` to probe whether the new branch can stay close to the accepted `A_k03` deployment shape.
- `configs/a_p03_medium_uhddeblur_gaussian_decoder_guidance_default_5h.toml`: first `5`-hour promotion run after the completed `A_m03/A_n03/A_o03` scouts; it keeps the `A_n03` default-strength decoder-guidance structure, expands the runtime tier to `300` minutes, and uses branch-local calibrated `batch_size=6`, `train_context_size=800`.
- `configs/a_q03_short_uhddeblur_gaussian_terminal_residual_probe.toml`: first short probe after the `A_p03` bottleneck diagnosis; it keeps the accepted decoder-guidance short geometry fixed but adds a lightweight Gaussian-guided terminal residual head to test more direct final-reconstruction control.
- `configs/a_r03_short_uhddeblur_gaussian_feature_refinement_probe.toml`: first post-`A_q03` short probe that keeps the accepted `A_n03` decoder-guidance geometry fixed but adds lightweight Gaussian-guided feature-space refinement before final RGB projection.
- `configs/a_s03_short_uhddeblur_edge_weighted_reconstruction_probe.toml`: supervision-only companion probe after `A_q03`; it keeps the accepted `A_n03` model fixed but adds a small edge-weighted reconstruction term to test whether the remaining gap is partly loss-side.
- `configs/a_t03_short_uhddeblur_feature_refinement_plus_edge_weighted_probe.toml`: combined post-`A_q03` short probe that enables both feature-space refinement and edge-weighted reconstruction to test whether the two ideas are complementary under the same lightweight budget.
- `configs/a_ab03_short_uhddeblur_edgeaware_feature_refinement_probe.toml`: first post-`A_aa03` structural redesign short probe; it keeps the accepted `M04/A_y03` short envelope fixed but replaces decoder self-detail hinting with an explicit high-resolution edge-aware prior inside Gaussian feature refinement.
- `configs/a_ac03_short_uhddeblur_confidence_normalized_feature_refinement_probe.toml`: post-`A_ab03` short probe that keeps the accepted `M04/A_y03` short envelope fixed but changes Gaussian feature refinement from uniform modulation into confidence-normalized modulation so hard regions receive more intervention than already-stable regions.
- `configs/a_ad03_short_uhddeblur_shared_prior_coupled_gaussian_probe.toml`: post-`A_ac03` companion short probe that keeps the accepted `M04/A_y03` short envelope fixed but introduces a shared compact prior-control stem consumed by both Gaussian decoder guidance and Gaussian feature refinement.
- `configs/a_b02_short_uhdhaze_patch_cache.toml`: `A_a02` follow-up branch for `UHD_haze` that keeps the same model and local patch-evaluation protocol but replaces repeated 4K JPEG online decode with a deterministic offline patch cache so the original high-VRAM runtime envelope can be recalibrated fairly.

Preflight check:

```bash
python3 scripts/train_v1_t01.py --config experiments/V1/templates/T01/configs/a01_short_uhdll_down4.toml --preflight
```
