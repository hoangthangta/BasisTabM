#!/usr/bin/env bash
# ==============================================================================================
# Bike Sharing: H=24 with calendar -- ALL experiments in one script:
# 81 configs x 3 seeds (0, 1, 2) = 243 runs.
#   (merged from run_bike_h24_cal_all.sh, run_bike_h24_cal_nofb_fb_norm.sh,
#    run_bike_h24_cal_tabm_do0.sh and run_bike_h24_cal_dlinear.sh; the NoFB/FB+Norm runs
#    were already in *_all.sh, so they appear once)
#   MLP (norm none/layer/batch)                         3
#   EfficientKAN                                         1
#   BSRBF-KAN (norm none/layer/batch)                    3
#   PatchTST (mix)                                       1
#   DLinear (univariate / mix / mix-individual)          3   lightweight reference, NOT budget-matched
#   TabM, dropout 0.0 (k = 2,4,8,16,32)  [used in paper] 5   notes do0_k*
#   TabM, dropout 0.1 (k = 2,4,8,16,32)  [old runs]      5   notes k* (library default 0.1, now explicit)
#   BasisTabM, 12 variants x 5 k                        60
# Output: output/bike/h24_cal/<model>/<file>__<note>.*
#
# BasisTabM variants (all: random-sign BatchEnsemble adapters, no norm1):
#   NoFB           (nofb_*_n2none)          : no basis at all              --basis_mode none --no-feature_basis
#   NoFB-Norm-BN/LN(nofb_*_n2batch|n2layer) : NoFB + BatchNorm / LayerNorm (norm2)
#   NoFB-Mix       (nofb_mix_*_n2none)      : Mix without input FB         --basis_mode mix --no-feature_basis
#   NoFB-MixNorm   (nofb_mix_*_n2batch|n2layer): NoFB-Mix + BatchNorm / LayerNorm
#   FB             (fb_*_n2none)            : input feature basis          --basis_mode none
#   FB-Norm-BN/LN  (fb_*_n2batch|n2layer)   : FB + BatchNorm / LayerNorm
#   FBMix          (fbmix_*_n2none)         : FB + basis branch in every layer --basis_mode mix
#   FBMixNorm      (fbmix_*_n2batch|n2layer): FBMix + BatchNorm / LayerNorm (norm2)
# All models except DLinear target the MLP budget (~93,624 params, within +-0.7%).
# NOTE: run_bike.py now passes dropout=args.dropout to TabM (default 0.0), so the old dropout-0.1
#       TabM runs need an explicit --dropout 0.1 to be reproduced; remove that block if not needed.
# ==============================================================================================

set -u
cd "$(dirname "$0")"

FAILED=()

# Common args for every run. Uses global $seed.
run() {
    python run_bike.py "$@" --lookback 24 --horizon 24 --calendar --epochs 50 \
        --seed "$seed" --overwrite || FAILED+=("seed=$seed $*")
    sleep 5s
}

# BasisTabM helper: btm <prefix> <k> <hidden> <norm2: ""|batch|layer> <basis_mode: none|mix> <feature_basis: 1|0>
btm() {
    local prefix=$1 k=$2 h=$3 n2=$4 bm=$5 fb=$6
    local fbflag=()
    [[ $fb == 0 ]] && fbflag=(--no-feature_basis)
    run --model basis_tabm --batch_size 64 --hidden_layers "$h" --tabm_k "$k" \
        --dropout 0.0 --activation silu --norm1_type "" --norm2_type "$n2" \
        --basis_mode "$bm" ${fbflag[@]+"${fbflag[@]}"} --scheduler OneCycleLR \
        --note "${prefix}_k${k}_h${h}_n1none_n2${n2:-none}_s${seed}"
}

KS=(2 4 8 16 32)

# Hidden sizes per k (index-aligned with KS). Param counts from the original script.
H_NOFB=(254 248 236 214 178)        # 93732 93792 93696 93456 93456
H_FB=(246 240 229 208 172)          # 93500 93512 93704 93704 93416
H_FBMIX=(245 237 224 199 159)       # 93676 93448 93784 93632 93408
H_FBMIX_BN=(242 233 215 185 141)    # 93540 93808 93696 93616 93648
H_FBMIX_LN=(244 236 222 198 159)    # 93796 93544 93444 93604 93726

# NoFB+Mix sizes: param counts verified from the finished runs.
H_NOFBMIX=(253 245 231 205 165)     # 93924 93760 93832 93480 93640
H_NOFBMIX_BN=(250 241 222 191 147)  # 93820 94184 93856 93656 94264
H_NOFBMIX_LN=(252 244 229 204 165)  # 94060 93872 93506 93464 93970

# NoFB / FB with hidden-layer normalisation (new; used-parameter counts computed with count_params)
H_NOFB_BN=(251 242 226 199 156)     # 93638 93496 93472 93704 93408
H_NOFB_LN=(252 246 235 213 178)     # 93504 93540 93782 93474 93812
H_FB_BN=(244 235 220 193 151)       # 93744 93532 93768 93760 93504
H_FB_LN=(245 239 228 207 172)       # 93624 93618 93776 93710 93760

for seed in 0 1 2; do

    # ===== MLP =====
    run --model mlp --hidden_layers 256 --batch_size 64 --norm_type batch --activation silu --scheduler OneCycleLR --note "h256_batch_s${seed}"   # 93624
    run --model mlp --hidden_layers 256 --batch_size 64 --norm_type ""    --activation silu --scheduler OneCycleLR --note "h256_none_s${seed}"    # 92440
    run --model mlp --hidden_layers 256 --batch_size 64 --norm_type layer --activation silu --scheduler OneCycleLR --note "h256_layer_s${seed}"   # 93624

    # ===== EfficientKAN =====
    run --model efficient_kan --batch_size 64 --hidden_layers 26 --grid_size 5 --scheduler OneCycleLR --note "h26_g5_s${seed}"   # 93600

    # ===== BSRBF-KAN =====
    run --model bsrbf_kan --batch_size 64 --hidden_layers 29 --grid_size 5 --norm_type ""    --scheduler OneCycleLR --note "h29_g5_none_s${seed}"    # 93960
    run --model bsrbf_kan --batch_size 64 --hidden_layers 29 --grid_size 5 --norm_type layer --scheduler OneCycleLR --note "h29_g5_layer_s${seed}"   # 94690
    run --model bsrbf_kan --batch_size 64 --hidden_layers 29 --grid_size 5 --norm_type batch --scheduler OneCycleLR --note "h29_g5_batch_s${seed}"   # 94690

    # ===== PatchTST (mix, batch 16; run_bike.py default scheduler = OneCycleLR) =====
    run --model patchtst --patchtst_exog mix --batch_size 16 --patch_len 8 --stride 4 --d_model 40 --n_heads 4 --e_layers 2 --d_ff 455 --note "mix_pl8_st4_dm40_ff455_s${seed}"

    # ===== DLinear (kernel 25; lightweight reference) =====
    run --model dlinear --dlinear_exog univariate --ma_kernel 25 --dlinear_individual 0 --batch_size 64 --scheduler OneCycleLR --note "uni_ma25_s${seed}"      # 1200
    run --model dlinear --dlinear_exog mix        --ma_kernel 25 --dlinear_individual 0 --batch_size 64 --scheduler OneCycleLR --note "mix_ma25_s${seed}"      # 1215
    run --model dlinear --dlinear_exog mix        --ma_kernel 25 --dlinear_individual 1 --batch_size 64 --scheduler OneCycleLR --note "mixind_ma25_s${seed}"   # 16815

    # ===== TabM, dropout = 0 (used in the paper) =====
    run --model tabm --hidden_layers 239 --tabm_k 2  --dropout 0.0 --batch_size 64 --scheduler OneCycleLR --note "do0_k2_h239_s${seed}"    # 93452
    run --model tabm --hidden_layers 210 --tabm_k 4  --dropout 0.0 --batch_size 64 --scheduler OneCycleLR --note "do0_k4_h210_s${seed}"    # 93840
    run --model tabm --hidden_layers 167 --tabm_k 8  --dropout 0.0 --batch_size 64 --scheduler OneCycleLR --note "do0_k8_h167_s${seed}"    # 93728
    run --model tabm --hidden_layers 117 --tabm_k 16 --dropout 0.0 --batch_size 64 --scheduler OneCycleLR --note "do0_k16_h117_s${seed}"   # 93744
    run --model tabm --hidden_layers 70  --tabm_k 32 --dropout 0.0 --batch_size 64 --scheduler OneCycleLR --note "do0_k32_h70_s${seed}"    # 93280

    # ===== TabM, dropout = 0.1 (old runs, library default; optional) =====
    run --model tabm --hidden_layers 239 --tabm_k 2  --dropout 0.1 --batch_size 64 --scheduler OneCycleLR --note "k2_h239_s${seed}"    # 93452
    run --model tabm --hidden_layers 210 --tabm_k 4  --dropout 0.1 --batch_size 64 --scheduler OneCycleLR --note "k4_h210_s${seed}"    # 93840
    run --model tabm --hidden_layers 167 --tabm_k 8  --dropout 0.1 --batch_size 64 --scheduler OneCycleLR --note "k8_h167_s${seed}"    # 93728
    run --model tabm --hidden_layers 117 --tabm_k 16 --dropout 0.1 --batch_size 64 --scheduler OneCycleLR --note "k16_h117_s${seed}"   # 93744
    run --model tabm --hidden_layers 70  --tabm_k 32 --dropout 0.1 --batch_size 64 --scheduler OneCycleLR --note "k32_h70_s${seed}"    # 93280

    # ===== BasisTabM =====
    for i in "${!KS[@]}"; do btm nofb     "${KS[i]}" "${H_NOFB[i]}"       ""    none 0; done
    for i in "${!KS[@]}"; do btm fb       "${KS[i]}" "${H_FB[i]}"         ""    none 1; done
    for i in "${!KS[@]}"; do btm fbmix    "${KS[i]}" "${H_FBMIX[i]}"      ""    mix  1; done
    for i in "${!KS[@]}"; do btm nofb_mix "${KS[i]}" "${H_NOFBMIX[i]}"    ""    mix  0; done
    for i in "${!KS[@]}"; do btm nofb_mix "${KS[i]}" "${H_NOFBMIX_BN[i]}" batch mix  0; done
    for i in "${!KS[@]}"; do btm nofb_mix "${KS[i]}" "${H_NOFBMIX_LN[i]}" layer mix  0; done
    for i in "${!KS[@]}"; do btm fbmix    "${KS[i]}" "${H_FBMIX_BN[i]}"   batch mix  1; done
    for i in "${!KS[@]}"; do btm fbmix    "${KS[i]}" "${H_FBMIX_LN[i]}"   layer mix  1; done
    for i in "${!KS[@]}"; do btm nofb     "${KS[i]}" "${H_NOFB_BN[i]}"    batch none 0; done
    for i in "${!KS[@]}"; do btm nofb     "${KS[i]}" "${H_NOFB_LN[i]}"    layer none 0; done
    for i in "${!KS[@]}"; do btm fb       "${KS[i]}" "${H_FB_BN[i]}"      batch none 1; done
    for i in "${!KS[@]}"; do btm fb       "${KS[i]}" "${H_FB_LN[i]}"      layer none 1; done

done

if ((${#FAILED[@]})); then
    echo "==== ${#FAILED[@]} run(s) failed ===="
    printf '  %s\n' "${FAILED[@]}"
fi
