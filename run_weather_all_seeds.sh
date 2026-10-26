#!/usr/bin/env bash
# ==============================================================================================
# Weather (OT, univariate): all models x 3 seeds (0, 1, 2)
# 190 configs (38 per group) x 3 seeds = 570 runs.
# Groups: A) multi-output L=96, H in {96, 192, 336, 720}; B) single-point L=144, H=1.
# MLP and BSRBF-KAN with norm = none / layer / batch.
# Output: output/weather/<model>/<file>__<note>__s<seed>.*   (note = L.._H.._mo|sp_<config>;
#         run_weather.py appends __s<seed> itself, so the seed is not in the note)
#
# BasisTabM variants (all: random-sign BatchEnsemble adapters, no norm1):
#   NoFB      (note *_nofb_*)          : no basis at all             --basis_mode none --no-feature_basis
#   FB        (note *_fb_*)            : input feature basis         --basis_mode none
#   FBMix     (note *_fbmix_*_n2none)  : FB + basis branch per layer --basis_mode mix
#   FBMixNorm (note *_fbmix_*_n2batch) : FBMix + BatchNorm (norm2)   --basis_mode mix --norm2_type batch
#   FBMixNorm (note *_fbmix_*_n2layer) : FBMix + LayerNorm (norm2)   [extra ablation]
# All BasisTabM models match the MLP budget of their group (within about +-0.5%).
# ==============================================================================================

set -u
cd "$(dirname "$0")"

for seed in 0 1 2; do

    # ##################### A) Multi-output, L=96, H=96 #####################

    # ===== MLP =====
    # mlp, norm=batch — 50208 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "batch" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h256_batch" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=none — 49504 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h256_none" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=layer — 50208 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "layer" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h256_layer" --overwrite --no_amp
    sleep 5s;

    # ===== EfficientKAN =====
    # efficient_kan — 49920 params
    python run_weather.py --model "efficient_kan" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h26_g5" --overwrite --no_amp
    sleep 5s;

    # ===== BSRBF-KAN =====
    # bsrbf_kan, norm=none — 50112 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h29_g5_none" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=layer — 50362 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h29_g5_layer" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=batch — 50362 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "batch" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_h29_g5_batch" --overwrite --no_amp
    sleep 5s;

    # ===== PatchTST =====
    # patchtst — 50200 params
    python run_weather.py --model "patchtst" --target "OT" --lookback 96 --horizon 96 --multi_output --batch_size 16 --patch_len 16 --stride 8 --d_model 28 --n_heads 4 --e_layers 2 --d_ff 90 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_pl16_st8_dm28_ff90" --overwrite --no_amp
    sleep 5s;

    # ===== TabM (k = 2, 4, 8, 16, 32) =====
    # tabm, k=2 — 50316 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "171" --tabm_k 2 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_k2_h171" --overwrite --no_amp
    sleep 5s;

    # tabm, k=4 — 50056 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "101" --tabm_k 4 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_k4_h101" --overwrite --no_amp
    sleep 5s;

    # tabm, k=8 — 49936 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "55" --tabm_k 8 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_k8_h55" --overwrite --no_amp
    sleep 5s;

    # tabm, k=16 — 49664 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "28" --tabm_k 16 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_k16_h28" --overwrite --no_amp
    sleep 5s;

    # tabm, k=32 — 51392 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "14" --tabm_k 32 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_k32_h14" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-NoFB: no basis at all (--basis_mode none --no-feature_basis), no norm =====
    # BasisTabM-NoFB, k=2 — 50274 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "251" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_nofb_k2_h251_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=4 — 50112 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "240" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_nofb_k4_h240_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=8 — 50256 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "222" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_nofb_k8_h222_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=16 — 50208 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "190" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_nofb_k16_h190_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=32 — 50112 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "142" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_nofb_k32_h142_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FB: input feature basis only (--basis_mode none), no norm =====
    # BasisTabM-FB, k=2 — 50258 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "247" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fb_k2_h247_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=4 — 50276 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "237" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fb_k4_h237_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=8 — 50168 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "218" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fb_k8_h218_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=16 — 50264 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "187" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fb_k16_h187_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=32 — 50312 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "140" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fb_k32_h140_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMix: FB + basis branch in every layer (--basis_mode mix), no norm =====
    # BasisTabM-FBMix, k=2 — 50148 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "243" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k2_h243_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=4 — 50160 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "230" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k4_h230_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=8 — 50232 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "207" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k8_h207_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=16 — 50216 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "169" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k16_h169_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=32 — 50248 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "116" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k32_h116_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm: FBMix + BatchNorm (--norm2_type batch) =====
    # BasisTabM-FBMixNorm (batch), k=2 — 50304 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "239" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k2_h239_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=4 — 50272 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "222" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k4_h222_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=8 — 50184 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "193" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k8_h193_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=16 — 50152 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "150" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k16_h150_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=32 — 50376 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "97" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k32_h97_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm (LayerNorm): FBMix + LayerNorm (--norm2_type layer), extra ablation =====
    # BasisTabM-FBMixNorm (layer), k=2 — 50230 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "241" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k2_h241_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=4 — 50200 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "228" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k4_h228_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=8 — 50194 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "205" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k8_h205_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=16 — 50296 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "168" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k16_h168_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=32 — 50158 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 96 --multi_output --hidden_layers "115" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H96_mo_fbmix_k32_h115_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # ##################### A) Multi-output, L=96, H=192 #####################

    # ===== MLP =====
    # mlp, norm=batch — 74880 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "batch" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h256_batch" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=none — 74176 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h256_none" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=layer — 74880 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "layer" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h256_layer" --overwrite --no_amp
    sleep 5s;

    # ===== EfficientKAN =====
    # efficient_kan — 74880 params
    python run_weather.py --model "efficient_kan" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h26_g5" --overwrite --no_amp
    sleep 5s;

    # ===== BSRBF-KAN =====
    # bsrbf_kan, norm=none — 75168 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h29_g5_none" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=layer — 75418 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h29_g5_layer" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=batch — 75418 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "batch" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_h29_g5_batch" --overwrite --no_amp
    sleep 5s;

    # ===== PatchTST =====
    # patchtst — 74846 params
    python run_weather.py --model "patchtst" --target "OT" --lookback 96 --horizon 192 --multi_output --batch_size 16 --patch_len 16 --stride 8 --d_model 24 --n_heads 4 --e_layers 2 --d_ff 139 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_pl16_st8_dm24_ff139" --overwrite --no_amp
    sleep 5s;

    # ===== TabM (k = 2, 4, 8, 16, 32) =====
    # tabm, k=2 — 75112 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "154" --tabm_k 2 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_k2_h154" --overwrite --no_amp
    sleep 5s;

    # tabm, k=4 — 75272 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "85" --tabm_k 4 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_k4_h85" --overwrite --no_amp
    sleep 5s;

    # tabm, k=8 — 74816 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "44" --tabm_k 8 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_k8_h44" --overwrite --no_amp
    sleep 5s;

    # tabm, k=16 — 75008 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "22" --tabm_k 16 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_k16_h22" --overwrite --no_amp
    sleep 5s;

    # tabm, k=32 — 72256 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "10" --tabm_k 32 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_k32_h10" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-NoFB: no basis at all (--basis_mode none --no-feature_basis), no norm =====
    # BasisTabM-NoFB, k=2 — 74754 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "251" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_nofb_k2_h251_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=4 — 74820 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "243" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_nofb_k4_h243_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=8 — 74976 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "228" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_nofb_k8_h228_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=16 — 74880 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "200" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_nofb_k16_h200_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=32 — 74880 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "155" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_nofb_k32_h155_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FB: input feature basis only (--basis_mode none), no norm =====
    # BasisTabM-FB, k=2 — 74942 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "249" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fb_k2_h249_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=4 — 74996 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "241" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fb_k4_h241_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=8 — 74816 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "225" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fb_k8_h225_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=16 — 74984 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "198" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fb_k16_h198_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=32 — 74888 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "153" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fb_k32_h153_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMix: FB + basis branch in every layer (--basis_mode mix), no norm =====
    # BasisTabM-FBMix, k=2 — 74940 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "246" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k2_h246_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=4 — 74912 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "235" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k4_h235_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=8 — 74968 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "215" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k8_h215_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=16 — 74920 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "180" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k16_h180_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=32 — 74760 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "126" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k32_h126_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm: FBMix + BatchNorm (--norm2_type batch) =====
    # BasisTabM-FBMixNorm (batch), k=2 — 75024 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "243" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k2_h243_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=4 — 74920 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "229" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k4_h229_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=8 — 74712 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "204" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k8_h204_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=16 — 74920 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "165" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k16_h165_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=32 — 74664 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "109" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k32_h109_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm (LayerNorm): FBMix + LayerNorm (--norm2_type layer), extra ablation =====
    # BasisTabM-FBMixNorm (layer), k=2 — 74836 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "244" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k2_h244_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=4 — 74770 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "233" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k4_h233_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=8 — 74754 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "213" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k8_h213_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=16 — 74926 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "179" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k16_h179_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=32 — 75012 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 192 --multi_output --hidden_layers "126" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H192_mo_fbmix_k32_h126_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # ##################### A) Multi-output, L=96, H=336 #####################

    # ===== MLP =====
    # mlp, norm=batch — 111888 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "batch" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h256_batch" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=none — 111184 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h256_none" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=layer — 111888 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "layer" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h256_layer" --overwrite --no_amp
    sleep 5s;

    # ===== EfficientKAN =====
    # efficient_kan — 112320 params
    python run_weather.py --model "efficient_kan" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h26_g5" --overwrite --no_amp
    sleep 5s;

    # ===== BSRBF-KAN =====
    # bsrbf_kan, norm=none — 112752 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h29_g5_none" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=layer — 113002 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h29_g5_layer" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=batch — 113002 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "batch" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_h29_g5_batch" --overwrite --no_amp
    sleep 5s;

    # ===== PatchTST =====
    # patchtst — 111856 params
    python run_weather.py --model "patchtst" --target "OT" --lookback 96 --horizon 336 --multi_output --batch_size 16 --patch_len 16 --stride 8 --d_model 24 --n_heads 4 --e_layers 2 --d_ff 92 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_pl16_st8_dm24_ff92" --overwrite --no_amp
    sleep 5s;

    # ===== TabM (k = 2, 4, 8, 16, 32) =====
    # tabm, k=2 — 112032 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "144" --tabm_k 2 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_k2_h144" --overwrite --no_amp
    sleep 5s;

    # tabm, k=4 — 111776 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "76" --tabm_k 4 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_k4_h76" --overwrite --no_amp
    sleep 5s;

    # tabm, k=8 — 112656 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "39" --tabm_k 8 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_k8_h39" --overwrite --no_amp
    sleep 5s;

    # tabm, k=16 — 111488 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "19" --tabm_k 16 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_k16_h19" --overwrite --no_amp
    sleep 5s;

    # tabm, k=32 — 112032 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "9" --tabm_k 32 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_k32_h9" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-NoFB: no basis at all (--basis_mode none --no-feature_basis), no norm =====
    # BasisTabM-NoFB, k=2 — 111912 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "252" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_nofb_k2_h252_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=4 — 111852 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "245" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_nofb_k4_h245_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=8 — 111936 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "232" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_nofb_k8_h232_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=16 — 111648 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "207" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_nofb_k16_h207_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=32 — 111696 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "165" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_nofb_k32_h165_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FB: input feature basis only (--basis_mode none), no norm =====
    # BasisTabM-FB, k=2 — 111812 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "250" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fb_k2_h250_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=4 — 111740 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "243" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fb_k4_h243_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=8 — 111800 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "230" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fb_k8_h230_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=16 — 111944 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "206" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fb_k16_h206_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=32 — 111944 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "164" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fb_k32_h164_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMix: FB + basis branch in every layer (--basis_mode mix), no norm =====
    # BasisTabM-FBMix, k=2 — 111668 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "247" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k2_h247_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=4 — 111824 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "238" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k4_h238_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=8 — 111704 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "220" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k8_h220_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=16 — 111720 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "188" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k16_h188_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=32 — 111768 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "135" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k32_h135_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm: FBMix + BatchNorm (--norm2_type batch) =====
    # BasisTabM-FBMixNorm (batch), k=2 — 111768 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "245" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k2_h245_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=4 — 111904 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "234" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k4_h234_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=8 — 111864 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "213" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k8_h213_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=16 — 111928 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "177" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k16_h177_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=32 — 111672 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "121" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k32_h121_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm (LayerNorm): FBMix + LayerNorm (--norm2_type layer), extra ablation =====
    # BasisTabM-FBMixNorm (layer), k=2 — 111720 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "246" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k2_h246_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=4 — 111850 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "237" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k4_h237_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=8 — 111678 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "219" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k8_h219_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=16 — 112096 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "188" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k16_h188_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=32 — 112038 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 336 --multi_output --hidden_layers "135" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H336_mo_fbmix_k32_h135_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # ##################### A) Multi-output, L=96, H=720 #####################

    # ===== MLP =====
    # mlp, norm=batch — 210576 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "batch" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h256_batch" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=none — 209872 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h256_none" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=layer — 210576 params
    python run_weather.py --model "mlp" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "256" --batch_size 64 --norm_type "layer" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h256_layer" --overwrite --no_amp
    sleep 5s;

    # ===== EfficientKAN =====
    # efficient_kan — 212160 params
    python run_weather.py --model "efficient_kan" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h26_g5" --overwrite --no_amp
    sleep 5s;

    # ===== BSRBF-KAN =====
    # bsrbf_kan, norm=none — 212976 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h29_g5_none" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=layer — 213226 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h29_g5_layer" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=batch — 213226 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "29" --grid_size 5 --norm_type "batch" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_h29_g5_batch" --overwrite --no_amp
    sleep 5s;

    # ===== PatchTST =====
    # patchtst — 210542 params
    python run_weather.py --model "patchtst" --target "OT" --lookback 96 --horizon 720 --multi_output --batch_size 16 --patch_len 16 --stride 8 --d_model 20 --n_heads 4 --e_layers 2 --d_ff 401 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_pl16_st8_dm20_ff401" --overwrite --no_amp
    sleep 5s;

    # ===== TabM (k = 2, 4, 8, 16, 32) =====
    # tabm, k=2 — 211072 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "136" --tabm_k 2 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_k2_h136" --overwrite --no_amp
    sleep 5s;

    # tabm, k=4 — 209160 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "69" --tabm_k 4 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_k4_h69" --overwrite --no_amp
    sleep 5s;

    # tabm, k=8 — 212048 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "35" --tabm_k 8 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_k8_h35" --overwrite --no_amp
    sleep 5s;

    # tabm, k=16 — 211072 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "17" --tabm_k 16 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_k16_h17" --overwrite --no_amp
    sleep 5s;

    # tabm, k=32 — 211712 params
    python run_weather.py --model "tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "8" --tabm_k 32 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_k32_h8" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-NoFB: no basis at all (--basis_mode none --no-feature_basis), no norm =====
    # BasisTabM-NoFB, k=2 — 210216 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "252" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_nofb_k2_h252_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=4 — 210660 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "247" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_nofb_k4_h247_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=8 — 210528 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "236" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_nofb_k8_h236_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=16 — 210336 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "215" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_nofb_k16_h215_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=32 — 210576 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "177" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_nofb_k32_h177_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FB: input feature basis only (--basis_mode none), no norm =====
    # BasisTabM-FB, k=2 — 210170 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "251" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fb_k2_h251_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=4 — 210608 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "246" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fb_k4_h246_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=8 — 210464 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "235" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fb_k8_h235_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=16 — 210248 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "214" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fb_k16_h214_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=32 — 210440 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "176" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fb_k32_h176_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMix: FB + basis branch in every layer (--basis_mode mix), no norm =====
    # BasisTabM-FBMix, k=2 — 210468 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "249" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k2_h249_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=4 — 210320 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "241" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k4_h241_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=8 — 210488 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "226" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k8_h226_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=16 — 210264 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "197" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k16_h197_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=32 — 210856 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "146" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k32_h146_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm: FBMix + BatchNorm (--norm2_type batch) =====
    # BasisTabM-FBMixNorm (batch), k=2 — 210636 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "248" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k2_h248_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=4 — 210568 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "239" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k4_h239_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=8 — 210648 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "222" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k8_h222_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=16 — 210184 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "190" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k16_h190_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=32 — 210120 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "136" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k32_h136_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm (LayerNorm): FBMix + LayerNorm (--norm2_type layer), extra ablation =====
    # BasisTabM-FBMixNorm (layer), k=2 — 210966 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "249" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k2_h249_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=4 — 210802 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "241" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k4_h241_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=8 — 210940 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "226" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k8_h226_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=16 — 210658 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "197" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k16_h197_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=32 — 210202 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 96 --horizon 720 --multi_output --hidden_layers "145" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L96_H720_mo_fbmix_k32_h145_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # ##################### B) Single-point, L=144, H=1 (predict t+1 only) #####################

    # ===== MLP =====
    # mlp, norm=batch — 38177 params
    python run_weather.py --model "mlp" --target "OT" --lookback 144 --horizon 1 --hidden_layers "256" --batch_size 64 --norm_type "batch" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h256_batch" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=none — 37377 params
    python run_weather.py --model "mlp" --target "OT" --lookback 144 --horizon 1 --hidden_layers "256" --batch_size 64 --norm_type "" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h256_none" --overwrite --no_amp
    sleep 5s;

    # mlp, norm=layer — 38177 params
    python run_weather.py --model "mlp" --target "OT" --lookback 144 --horizon 1 --hidden_layers "256" --batch_size 64 --norm_type "layer" --activation "silu" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h256_layer" --overwrite --no_amp
    sleep 5s;

    # ===== EfficientKAN =====
    # efficient_kan — 37700 params
    python run_weather.py --model "efficient_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "26" --grid_size 5 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h26_g5" --overwrite --no_amp
    sleep 5s;

    # ===== BSRBF-KAN =====
    # bsrbf_kan, norm=none — 37845 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "29" --grid_size 5 --norm_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h29_g5_none" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=layer — 38191 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "29" --grid_size 5 --norm_type "layer" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h29_g5_layer" --overwrite --no_amp
    sleep 5s;

    # bsrbf_kan, norm=batch — 38191 params
    python run_weather.py --model "bsrbf_kan" --target "OT" --lookback 144 --horizon 1 --hidden_layers "29" --grid_size 5 --norm_type "batch" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_h29_g5_batch" --overwrite --no_amp
    sleep 5s;

    # ===== PatchTST =====
    # patchtst — 38179 params
    python run_weather.py --model "patchtst" --target "OT" --lookback 144 --horizon 1 --batch_size 16 --patch_len 8 --stride 4 --d_model 44 --n_heads 4 --e_layers 2 --d_ff 103 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_pl8_st4_dm44_ff103" --overwrite --no_amp
    sleep 5s;

    # ===== TabM (k = 2, 4, 8, 16, 32) =====
    # tabm, k=2 — 38240 params
    python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "253" --tabm_k 2 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_k2_h253" --overwrite --no_amp
    sleep 5s;

    # tabm, k=4 — 38176 params
    python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "241" --tabm_k 4 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_k4_h241" --overwrite --no_amp
    sleep 5s;

    # tabm, k=8 — 38120 params
    python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "220" --tabm_k 8 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_k8_h220" --overwrite --no_amp
    sleep 5s;

    # tabm, k=16 — 38224 params
    python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "187" --tabm_k 16 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_k16_h187" --overwrite --no_amp
    sleep 5s;

    # tabm, k=32 — 38240 params
    python run_weather.py --model "tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "140" --tabm_k 32 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_k32_h140" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-NoFB: no basis at all (--basis_mode none --no-feature_basis), no norm =====
    # BasisTabM-NoFB, k=2 — 38193 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "251" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_nofb_k2_h251_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=4 — 38107 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "239" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_nofb_k4_h239_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=8 — 38179 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "219" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_nofb_k8_h219_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=16 — 38234 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "186" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_nofb_k16_h186_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-NoFB, k=32 — 38171 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "139" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_nofb_k32_h139_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FB: input feature basis only (--basis_mode none), no norm =====
    # BasisTabM-FB, k=2 — 38145 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "243" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fb_k2_h243_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=4 — 38168 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "232" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fb_k4_h232_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=8 — 38156 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "212" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fb_k8_h212_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=16 — 38236 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "180" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fb_k16_h180_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FB, k=32 — 38126 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "134" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fb_k32_h134_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMix: FB + basis branch in every layer (--basis_mode mix), no norm =====
    # BasisTabM-FBMix, k=2 — 38178 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "240" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k2_h240_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=4 — 38142 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "226" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k4_h226_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=8 — 38106 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "202" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k8_h202_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=16 — 38238 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "166" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k16_h166_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMix, k=32 — 38142 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "118" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k32_h118_n1none_n2none" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm: FBMix + BatchNorm (--norm2_type batch) =====
    # BasisTabM-FBMixNorm (batch), k=2 — 38196 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "234" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k2_h234_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=4 — 38260 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "216" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k4_h216_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=8 — 38250 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "186" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k8_h186_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=16 — 38248 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "144" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k16_h144_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=32 — 38280 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "96" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k32_h96_n1none_n2batch" --overwrite --no_amp
    sleep 5s;

    # ===== BasisTabM-FBMixNorm (LayerNorm): FBMix + LayerNorm (--norm2_type layer), extra ablation =====
    # BasisTabM-FBMixNorm (layer), k=2 — 38193 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "237" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k2_h237_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=4 — 38105 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "223" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k4_h223_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=8 — 38152 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "200" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k8_h200_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=16 — 38148 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "164" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k16_h164_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=32 — 38103 params
    python run_weather.py --model "basis_tabm" --target "OT" --lookback 144 --horizon 1 --hidden_layers "117" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --note "L144_H1_sp_fbmix_k32_h117_n1none_n2layer" --overwrite --no_amp
    sleep 5s;

done
