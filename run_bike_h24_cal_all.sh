#!/usr/bin/env bash
# ==============================================================================================
# Bike Sharing: H=24 with calendar, 73 configs x 3 seeds (0, 1, 2) = 219 runs.
#   MLP (norm none/layer/batch)            3
#   EfficientKAN                            1
#   BSRBF-KAN (norm none/layer/batch)       3
#   PatchTST                                1
#   TabM (k = 2,4,8,16,32)                  5
#   BasisTabM, 12 variants x 5 k           60
# Output: output/bike/h24_cal/<model>/<file>__<note>.*
#
# BasisTabM variants (all: random-sign BatchEnsemble adapters, no norm1):
#   NoFB           (nofb_*)                 : no basis at all              --basis_mode none --no-feature_basis
#   FB             (fb_*)                   : input feature basis          --basis_mode none
#   FBMix          (fbmix_*_n2none)         : FB + basis branch per layer  --basis_mode mix
#   FBMixNorm-BN   (fbmix_*_n2batch)        : FBMix + BatchNorm (norm2)    --basis_mode mix --norm2_type batch
#   FBMixNorm-LN   (fbmix_*_n2layer)        : FBMix + LayerNorm (norm2)    [extra ablation]
#   NoFB-Mix       (nofb_mix_*_n2none)      : Mix without input FB         --basis_mode mix --no-feature_basis
#   NoFB-MixNorm-BN(nofb_mix_*_n2batch)     : NoFB-Mix + BatchNorm
#   NoFB-MixNorm-LN(nofb_mix_*_n2layer)     : NoFB-Mix + LayerNorm
#   NoFB-Norm-BN   (nofb_*_n2batch)         : NoFB + BatchNorm             --basis_mode none --no-feature_basis --norm2_type batch
#   NoFB-Norm-LN   (nofb_*_n2layer)         : NoFB + LayerNorm
#   FB-Norm-BN     (fb_*_n2batch)           : FB + BatchNorm               --basis_mode none --norm2_type batch
#   FB-Norm-LN     (fb_*_n2layer)           : FB + LayerNorm
# All BasisTabM models target the MLP budget (~93,624 params, within +-0.7%).
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

    # ===== PatchTST ===== (batch 16, default scheduler — intentional?; param count to re-check)
    run --model patchtst --patchtst_exog mix --batch_size 16 --patch_len 8 --stride 4 --d_model 40 --n_heads 4 --e_layers 2 --d_ff 455 --note "mix_pl8_st4_dm40_ff455_s${seed}"

    # ===== TabM =====
    run --model tabm --hidden_layers 239 --tabm_k 2  --batch_size 64 --scheduler OneCycleLR --note "k2_h239_s${seed}"    # 93452
    run --model tabm --hidden_layers 210 --tabm_k 4  --batch_size 64 --scheduler OneCycleLR --note "k4_h210_s${seed}"    # 93840
    run --model tabm --hidden_layers 167 --tabm_k 8  --batch_size 64 --scheduler OneCycleLR --note "k8_h167_s${seed}"    # 93728
    run --model tabm --hidden_layers 117 --tabm_k 16 --batch_size 64 --scheduler OneCycleLR --note "k16_h117_s${seed}"   # 93744
    run --model tabm --hidden_layers 70  --tabm_k 32 --batch_size 64 --scheduler OneCycleLR --note "k32_h70_s${seed}"    # 93280

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
