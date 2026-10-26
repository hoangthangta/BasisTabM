#!/usr/bin/env bash
# ==============================================================================================
# Bike Sharing: H=24 with calendar, 38 configs x 3 seeds (0, 1, 2) = 114 runs.
# MLP and BSRBF-KAN with norm = none / layer / batch.
# Output: output/bike/h24_cal/<model>/<file>__<note>.*
#
# BasisTabM variants (all: random-sign BatchEnsemble adapters, no norm1):
#   NoFB      (note nofb_*)          : no basis at all            --basis_mode none --no-feature_basis
#   FB        (note fb_*)            : input feature basis        --basis_mode none
#   FBMix     (note fbmix_*_n2none)  : FB + basis branch per layer --basis_mode mix
#   FBMixNorm (note fbmix_*_n2batch) : FBMix + BatchNorm (norm2)  --basis_mode mix --norm2_type batch
#   FBMixNorm (note fbmix_*_n2layer) : FBMix + LayerNorm (norm2)  [extra ablation]
# All BasisTabM models match the MLP budget (~93,624 params, within +-0.3%).
# ==============================================================================================

set -u
cd "$(dirname "$0")"

for seed in 0 1 2; do

    # ===== MLP =====
    # mlp, norm=batch — 93624 params
    python run_bike.py --model "mlp" --lookback "24" --hidden_layers "256" --batch_size "64" --epochs "50" --norm_type "batch" --activation "silu" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h256_batch_s${seed}" --overwrite
    sleep 5s;

    # mlp, norm=none — 92440 params
    python run_bike.py --model "mlp" --lookback "24" --hidden_layers "256" --batch_size "64" --epochs "50" --norm_type "" --activation "silu" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h256_none_s${seed}" --overwrite
    sleep 5s;

    # mlp, norm=layer — 93624 params
    python run_bike.py --model "mlp" --lookback "24" --hidden_layers "256" --batch_size "64" --epochs "50" --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h256_layer_s${seed}" --overwrite
    sleep 5s;

    # ===== EfficientKAN =====
    # efficient_kan — 93600 params
    python run_bike.py --model "efficient_kan" --lookback "24" --batch_size "64" --hidden_layers "26" --grid_size "5" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h26_g5_s${seed}" --overwrite
    sleep 5s;

    # ===== BSRBF-KAN =====
    # bsrbf_kan, norm=none — 93960 params
    python run_bike.py --model "bsrbf_kan" --lookback "24" --batch_size "64" --hidden_layers "29" --grid_size "5" --norm_type "" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h29_g5_none_s${seed}" --overwrite
    sleep 5s;

    # bsrbf_kan, norm=layer — 94690 params
    python run_bike.py --model "bsrbf_kan" --lookback "24" --batch_size "64" --hidden_layers "29" --grid_size "5" --norm_type "layer" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h29_g5_layer_s${seed}" --overwrite
    sleep 5s;

    # bsrbf_kan, norm=batch — 94690 params
    python run_bike.py --model "bsrbf_kan" --lookback "24" --batch_size "64" --hidden_layers "29" --grid_size "5" --norm_type "batch" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "h29_g5_batch_s${seed}" --overwrite
    sleep 5s;

    # ===== PatchTST =====
    # patchtst — 93614 + 15 (re-check) params
    python run_bike.py --model "patchtst" --patchtst_exog "mix" --lookback "24" --batch_size "16" --epochs "50" --seed "$seed" --patch_len "8" --stride "4" --d_model "40" --n_heads "4" --e_layers "2" --d_ff "455" --horizon "24" --calendar --note "mix_pl8_st4_dm40_ff455_s${seed}" --overwrite
    sleep 5s;

    # ===== TabM (k = 2, 4, 8, 16, 32) =====
    # tabm, k=2 — 93452 params
    python run_bike.py --model "tabm" --lookback 24 --hidden_layers "239" --tabm_k 2 --batch_size 64 --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "k2_h239_s${seed}" --overwrite
    sleep 5s;

    # tabm — 93840 params
    python run_bike.py --model "tabm" --lookback "24" --hidden_layers "210" --tabm_k "4" --batch_size "64" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "k4_h210_s${seed}" --overwrite
    sleep 5s;

    # tabm — 93728 params
    python run_bike.py --model "tabm" --lookback "24" --hidden_layers "167" --tabm_k "8" --batch_size "64" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "k8_h167_s${seed}" --overwrite
    sleep 5s;

    # tabm — 93744 params
    python run_bike.py --model "tabm" --lookback "24" --hidden_layers "117" --tabm_k "16" --batch_size "64" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "k16_h117_s${seed}" --overwrite
    sleep 5s;

    # tabm — 93280 params
    python run_bike.py --model "tabm" --lookback "24" --hidden_layers "70" --tabm_k "32" --batch_size "64" --epochs "50" --scheduler "OneCycleLR" --seed "$seed" --horizon "24" --calendar --note "k32_h70_s${seed}" --overwrite
    sleep 5s;

    # ===== BasisTabM-NoFB: no basis at all (--basis_mode none --no-feature_basis), no norm =====
    # BasisTabM-NoFB, k=2 — 93732 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "254" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "nofb_k2_h254_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-NoFB, k=4 — 93792 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "248" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "nofb_k4_h248_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-NoFB, k=8 — 93696 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "236" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "nofb_k8_h236_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-NoFB, k=16 — 93456 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "214" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "nofb_k16_h214_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-NoFB, k=32 — 93456 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "178" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "none" --no-feature_basis --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "nofb_k32_h178_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # ===== BasisTabM-FB: input feature basis only (--basis_mode none), no norm =====
    # BasisTabM-FB, k=2 — 93500 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "246" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fb_k2_h246_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FB, k=4 — 93512 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "240" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fb_k4_h240_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FB, k=8 — 93704 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "229" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fb_k8_h229_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FB, k=16 — 93704 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "208" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fb_k16_h208_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FB, k=32 — 93416 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "172" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fb_k32_h172_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # ===== BasisTabM-FBMix: FB + basis branch in every layer (--basis_mode mix), no norm =====
    # BasisTabM-FBMix, k=2 — 93676 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "245" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k2_h245_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMix, k=4 — 93448 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "237" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k4_h237_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMix, k=8 — 93784 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "224" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k8_h224_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMix, k=16 — 93632 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "199" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k16_h199_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMix, k=32 — 93408 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "159" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k32_h159_n1none_n2none_s${seed}" --overwrite
    sleep 5s;

    # ===== BasisTabM-FBMixNorm: FBMix + BatchNorm (--norm2_type batch) =====
    # BasisTabM-FBMixNorm (batch), k=2 — 93540 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "242" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k2_h242_n1none_n2batch_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=4 — 93808 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "233" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k4_h233_n1none_n2batch_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=8 — 93696 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "215" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k8_h215_n1none_n2batch_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=16 — 93616 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "185" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k16_h185_n1none_n2batch_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (batch), k=32 — 93648 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "141" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "batch" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k32_h141_n1none_n2batch_s${seed}" --overwrite
    sleep 5s;

    # ===== BasisTabM-FBMixNorm (LayerNorm): FBMix + LayerNorm (--norm2_type layer), extra ablation =====
    # BasisTabM-FBMixNorm (layer), k=2 — 93796 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "244" --tabm_k 2 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k2_h244_n1none_n2layer_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=4 — 93544 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "236" --tabm_k 4 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k4_h236_n1none_n2layer_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=8 — 93444 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "222" --tabm_k 8 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k8_h222_n1none_n2layer_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=16 — 93604 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "198" --tabm_k 16 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k16_h198_n1none_n2layer_s${seed}" --overwrite
    sleep 5s;

    # BasisTabM-FBMixNorm (layer), k=32 — 93726 params
    python run_bike.py --model "basis_tabm" --lookback 24 --batch_size 64 --hidden_layers "159" --tabm_k 32 --dropout 0.0 --activation "silu" --norm1_type "" --norm2_type "layer" --basis_mode "mix" --epochs 50 --scheduler "OneCycleLR" --seed "$seed" --horizon 24 --calendar --note "fbmix_k32_h159_n1none_n2layer_s${seed}" --overwrite
    sleep 5s;

done
