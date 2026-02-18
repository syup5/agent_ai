#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
3. 어텐션(Attention) 메커니즘 시각화
=============================================================================
Transformer의 핵심인 Self-Attention이 어떻게 작동하는지 시각화합니다.

- Query, Key, Value 개념 설명
- Attention 가중치 히트맵
- Multi-Head Attention 시각화

실행: conda activate agent && python 20260215_160921_03_attention.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

def setup_korean_font():
    import subprocess
    try:
        result = subprocess.run(
            ['fc-match', '-f', '%{file}', 'Noto Sans CJK KR'],
            capture_output=True, text=True, timeout=5)
        font_path = result.stdout.strip()
        if font_path and os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            print(f"[INFO] 한글 폰트: {prop.get_name()} ({font_path})")
            plt.rcParams['axes.unicode_minus'] = False
            return
    except Exception:
        pass
    for name in ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans CJK JP']:
        if name in [f.name for f in font_manager.fontManager.ttflist]:
            plt.rcParams['font.family'] = name
            print(f"[INFO] 한글 폰트 (fallback): {name}")
            break
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Part A: Self-Attention 수학적 과정 시각화
# =============================================================================
def visualize_self_attention_math():
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    이 수식의 각 단계를 시각화합니다.
    """
    np.random.seed(42)

    # 간단한 예시: 4개 토큰, 차원 3
    tokens = ["The", "cat", "sat", "down"]
    d_k = 3
    seq_len = len(tokens)

    # 임베딩 벡터 (임의 생성)
    X = np.random.randn(seq_len, d_k)

    # Q, K, V 가중치 행렬
    W_Q = np.random.randn(d_k, d_k) * 0.5
    W_K = np.random.randn(d_k, d_k) * 0.5
    W_V = np.random.randn(d_k, d_k) * 0.5

    # Q, K, V 계산
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # Attention score 계산
    scores = Q @ K.T / np.sqrt(d_k)

    # Softmax
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    attn_weights = softmax(scores)

    # 결과
    output = attn_weights @ V

    # ── 시각화 (6단계) ──
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Self-Attention 메커니즘 단계별 시각화\n"
                 "Attention(Q,K,V) = softmax(QK^T / √d_k) · V",
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. 입력 임베딩 X
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(X, cmap='coolwarm', aspect='auto')
    ax1.set_title("① 입력 임베딩 X", fontsize=12, fontweight='bold')
    ax1.set_yticks(range(seq_len))
    ax1.set_yticklabels(tokens)
    ax1.set_xlabel("차원")
    for i in range(seq_len):
        for j in range(d_k):
            ax1.text(j, i, f"{X[i,j]:.2f}", ha='center', va='center', fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.7)

    # 2. Q, K, V 행렬
    ax2 = fig.add_subplot(2, 3, 2)
    combined = np.hstack([Q, np.full((seq_len, 1), np.nan),
                          K, np.full((seq_len, 1), np.nan), V])
    im2 = ax2.imshow(combined, cmap='coolwarm', aspect='auto')
    ax2.set_title("② Q | K | V 행렬", fontsize=12, fontweight='bold')
    ax2.set_yticks(range(seq_len))
    ax2.set_yticklabels(tokens)
    ax2.axvline(x=2.5, color='black', linewidth=2)
    ax2.axvline(x=6.5, color='black', linewidth=2)
    ax2.text(1, -0.7, "Q", ha='center', fontsize=12, fontweight='bold', color='red')
    ax2.text(5, -0.7, "K", ha='center', fontsize=12, fontweight='bold', color='blue')
    ax2.text(9, -0.7, "V", ha='center', fontsize=12, fontweight='bold', color='green')

    # 3. QK^T (Attention Scores)
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(scores, cmap='YlOrRd', aspect='auto')
    ax3.set_title("③ QK^T / √d_k (Attention Scores)", fontsize=12, fontweight='bold')
    ax3.set_xticks(range(seq_len))
    ax3.set_yticks(range(seq_len))
    ax3.set_xticklabels(tokens)
    ax3.set_yticklabels(tokens)
    ax3.set_xlabel("Key (참조되는 토큰)")
    ax3.set_ylabel("Query (질의하는 토큰)")
    for i in range(seq_len):
        for j in range(seq_len):
            ax3.text(j, i, f"{scores[i,j]:.2f}", ha='center', va='center', fontsize=9)
    fig.colorbar(im3, ax=ax3, shrink=0.7)

    # 4. Softmax → Attention Weights
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(attn_weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax4.set_title("④ Softmax → Attention 가중치", fontsize=12, fontweight='bold')
    ax4.set_xticks(range(seq_len))
    ax4.set_yticks(range(seq_len))
    ax4.set_xticklabels(tokens)
    ax4.set_yticklabels(tokens)
    ax4.set_xlabel("Key")
    ax4.set_ylabel("Query")
    for i in range(seq_len):
        for j in range(seq_len):
            ax4.text(j, i, f"{attn_weights[i,j]:.2f}", ha='center', va='center',
                     fontsize=9, color='white' if attn_weights[i,j] > 0.5 else 'black')
    fig.colorbar(im4, ax=ax4, shrink=0.7)

    # 5. Attention 가중치 × V
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(output, cmap='coolwarm', aspect='auto')
    ax5.set_title("⑤ Attention(Q,K,V) 최종 출력", fontsize=12, fontweight='bold')
    ax5.set_yticks(range(seq_len))
    ax5.set_yticklabels(tokens)
    ax5.set_xlabel("차원")
    for i in range(seq_len):
        for j in range(d_k):
            ax5.text(j, i, f"{output[i,j]:.2f}", ha='center', va='center', fontsize=9)
    fig.colorbar(im5, ax=ax5, shrink=0.7)

    # 6. 설명 텍스트
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    explanation = (
        "Self-Attention 핵심 개념:\n\n"
        "• Query(Q): \"나는 무엇을 찾고 있는가?\"\n"
        "  각 토큰이 다른 토큰에게 묻는 질문\n\n"
        "• Key(K): \"나는 어떤 정보를 가지고 있는가?\"\n"
        "  각 토큰이 자신을 설명하는 라벨\n\n"
        "• Value(V): \"내가 제공할 수 있는 실제 정보\"\n"
        "  실제로 전달되는 내용\n\n"
        "• QK^T: Q와 K의 내적으로 유사도 계산\n"
        "• softmax: 유사도를 확률로 변환\n"
        "• 최종 출력: V의 가중 평균"
    )
    ax6.text(0.05, 0.95, explanation, fontsize=11, va='top',
             transform=ax6.transAxes, linespacing=1.6,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='orange'))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUTPUT_DIR, "03_self_attention_math.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part B: 실제 GPT-2 모델의 Attention 가중치 시각화
# =============================================================================
def visualize_real_attention():
    """GPT-2 모델에서 실제 attention 가중치를 추출하여 시각화"""
    from transformers import GPT2Model, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    model.eval()

    text = "The cat sat on the mat and looked at the bird"
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # attentions: tuple of (batch, heads, seq_len, seq_len) per layer
    attentions = outputs.attentions

    # 첫 번째 레이어, 4개 헤드 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'GPT-2 실제 Attention 가중치 (Layer 0)\n입력: "{text}"',
                 fontsize=16, fontweight='bold', y=0.98)

    layer_attn = attentions[0][0].numpy()  # (num_heads, seq_len, seq_len)

    for head_idx, ax in enumerate(axes.flat):
        attn = layer_attn[head_idx]
        im = ax.imshow(attn, cmap='Blues', vmin=0)
        ax.set_title(f"Head {head_idx}", fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel("Key (참조)")
        ax.set_ylabel("Query (질의)")
        fig.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUTPUT_DIR, "03_real_attention_heads.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part C: Attention 흐름 (어떤 단어가 어떤 단어에 주목하는지)
# =============================================================================
def visualize_attention_flow():
    """특정 토큰이 다른 토큰에 얼마나 주목하는지 연결선으로 시각화"""
    from transformers import GPT2Model, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    model.eval()

    text = "The dog chased the cat"
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # 모든 헤드의 평균 attention (Layer 0)
    attn = outputs.attentions[0][0].mean(dim=0).numpy()  # (seq_len, seq_len)
    n = len(tokens)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Attention 흐름 시각화 (GPT-2 Layer 0, 전체 헤드 평균)\n입력: "{text}"',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    # 좌측: Query 토큰, 우측: Key 토큰
    left_x, right_x = 0.15, 0.85
    y_positions = np.linspace(0.85, 0.15, n)

    for i, tok in enumerate(tokens):
        # 왼쪽 (Query)
        ax.text(left_x, y_positions[i], tok.replace('Ġ', '_ '),
                fontsize=13, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F0FE',
                          edgecolor='#4A90D9', linewidth=2),
                transform=ax.transAxes)
        # 오른쪽 (Key)
        ax.text(right_x, y_positions[i], tok.replace('Ġ', '_ '),
                fontsize=13, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3E8',
                          edgecolor='#D9904A', linewidth=2),
                transform=ax.transAxes)

        # 연결선 (attention 가중치에 비례하는 두께/투명도)
        for j in range(n):
            weight = attn[i, j]
            if weight > 0.05:  # 임계값 이상만 표시
                ax.plot([left_x + 0.06, right_x - 0.06],
                        [y_positions[i], y_positions[j]],
                        color='#E74C3C', alpha=min(weight * 2, 1.0),
                        linewidth=weight * 8, transform=ax.transAxes)

    ax.text(left_x, 0.95, "Query\n(질의)", fontsize=12, ha='center',
            fontweight='bold', color='#4A90D9', transform=ax.transAxes)
    ax.text(right_x, 0.95, "Key\n(참조)", fontsize=12, ha='center',
            fontweight='bold', color='#D9904A', transform=ax.transAxes)

    ax.text(0.5, 0.03, "선의 굵기/투명도 = Attention 가중치 (높을수록 더 많이 주목)",
            fontsize=11, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_attention_flow.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  3. 어텐션(Attention) 메커니즘 시각화")
    print("=" * 60)
    print("\n[A] Self-Attention 수학적 과정...")
    visualize_self_attention_math()
    print("\n[B] GPT-2 실제 Attention 가중치...")
    visualize_real_attention()
    print("\n[C] Attention 흐름 시각화...")
    visualize_attention_flow()
    print(f"\n완료! 이미지: {OUTPUT_DIR}")
