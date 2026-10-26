#!/usr/bin/env bash
# ===================================================================================================
# Weather (OT, univariate), multi-output: lookback L=96, horizon H=96 (16 h ahead).
# 38 configs x 3 seeds (0, 1, 2) = 114 runs.
# MLP and BSRBF-KAN with norm = none / layer / batch.
# Output: output/weather/<model>/<file>__<note>__s<seed>.*   (note = L96_H96_mo_<config>;
#         run_weather.py appends __s<seed> itself, so the seed is not in the note)
#
# BasisTabM variants (all: random-sign BatchEnsemble adapters, no norm1):
#   NoFB      (note *_nofb_*)          : no basis at all             --basis_mode none --no-feature_basis
#   FB        (note *_fb_*)            : input feature basis         --basis_mode none
#   FBMix     (note *_fbmix_*_n2none)  : FB + basis branch per layer --basis_mode mix
#   FBMixNorm (note *_fbmix_*_n2batch) : FBMix + BatchNorm (norm2)   --basis_mode mix --norm2_type batch
#   FBMixNorm (note *_fbmix_*_n2layer) : FBMix + LayerNorm (norm2)   [extra ablation]
# All BasisTabM models match the MLP budget (~50,208 params, within +-0.5%).
# ====================================================================================================

set -u
cd "$(dirname "$0")"

for seed in 0 1 2; do

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

done
