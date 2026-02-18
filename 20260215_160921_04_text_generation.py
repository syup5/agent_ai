#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
4. 텍스트 생성(Text Generation) 과정 시각화
=============================================================================
LLM이 어떻게 다음 토큰을 예측하고 문장을 생성하는지 시각화합니다.

- 다음 토큰 확률 분포
- Temperature, Top-k, Top-p 샘플링 차이
- 자기회귀(Autoregressive) 생성 과정

실행: conda activate agent && python 20260215_160921_04_text_generation.py
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
# Part A: 다음 토큰 예측 확률 분포
# =============================================================================
def visualize_next_token_prediction():
    """GPT-2가 다음 토큰을 예측할 때의 확률 분포를 시각화"""
    from transformers import GPT2LMHeadModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # 마지막 토큰 위치의 로짓 → 확률
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    probs = torch.softmax(logits, dim=-1)

    # 상위 20개 토큰
    top_k = 20
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'다음 토큰 예측 확률 분포\n입력: "{prompt} ___"',
                 fontsize=16, fontweight='bold', y=0.98)

    # 바 차트 - 상위 20개
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, top_k))
    bars = ax.barh(range(top_k-1, -1, -1), top_probs.numpy(), color=colors)
    ax.set_yticks(range(top_k-1, -1, -1))
    ax.set_yticklabels([f'"{t.strip()}"' for t in top_tokens], fontsize=10)
    ax.set_xlabel("확률", fontsize=12)
    ax.set_title(f"상위 {top_k}개 예측 토큰", fontsize=13, fontweight='bold')

    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(bar.get_width() + 0.005, top_k-1-i,
                f"{prob:.3f} ({prob*100:.1f}%)",
                va='center', fontsize=9)

    # 파이 차트 - 상위 5개 + 나머지
    ax2 = axes[1]
    top5_probs = top_probs[:5].numpy()
    top5_tokens = [t.strip() for t in top_tokens[:5]]
    rest_prob = 1.0 - top5_probs.sum()

    sizes = list(top5_probs) + [rest_prob]
    labels = [f'"{t}" ({p:.1%})' for t, p in zip(top5_tokens, top5_probs)]
    labels.append(f'기타 50,252개\n({rest_prob:.1%})')

    pie_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#95A5A6']
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=pie_colors,
        autopct='', startangle=90, pctdistance=0.85)

    for text in texts:
        text.set_fontsize(10)
    ax2.set_title("확률 분포 (상위 5개 vs 나머지)", fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUTPUT_DIR, "04_next_token_prediction.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part B: Temperature, Top-k, Top-p 샘플링 비교
# =============================================================================
def visualize_sampling_strategies():
    """
    Temperature, Top-k, Top-p 가 확률 분포에 미치는 영향을 시각화

    - Temperature: 분포의 '날카로움' 조절 (낮을수록 확정적)
    - Top-k: 상위 k개만 후보로 남기고 나머지 확률을 0으로
    - Top-p (Nucleus): 누적 확률이 p가 될 때까지만 후보로 남김
    """
    np.random.seed(42)

    # 가상의 로짓 (10개 토큰)
    token_labels = ["Paris", "the", "Lyon", "a", "known", "famous",
                    "Berlin", "city", "where", "not"]
    raw_logits = np.array([4.5, 2.8, 2.3, 1.9, 1.5, 1.2, 0.8, 0.5, 0.2, -0.5])

    def softmax_with_temp(logits, temperature=1.0):
        scaled = logits / temperature
        e = np.exp(scaled - np.max(scaled))
        return e / e.sum()

    def top_k_filter(probs, k):
        result = np.zeros_like(probs)
        top_idx = np.argsort(probs)[-k:]
        result[top_idx] = probs[top_idx]
        return result / result.sum()

    def top_p_filter(probs, p):
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, p) + 1
        result = np.zeros_like(probs)
        result[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
        return result / result.sum()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("샘플링 전략 비교: Temperature / Top-k / Top-p",
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Temperature 비교 ──
    temps = [0.3, 1.0, 2.0]
    for i, temp in enumerate(temps):
        ax = axes[0, i]
        probs = softmax_with_temp(raw_logits, temp)
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(probs)))
        bars = ax.bar(range(len(probs)), probs, color=colors)
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("확률")
        ax.set_title(f"Temperature = {temp}", fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.0)

        # 최대 확률 표시
        max_idx = np.argmax(probs)
        ax.text(max_idx, probs[max_idx] + 0.02, f"{probs[max_idx]:.2f}",
                ha='center', fontsize=10, fontweight='bold', color='red')

        if temp == 0.3:
            ax.text(0.5, 0.95, "← 확정적 (거의 Greedy)", fontsize=9,
                    ha='center', transform=ax.transAxes, color='blue')
        elif temp == 2.0:
            ax.text(0.5, 0.95, "← 창의적 (분포 평탄화)", fontsize=9,
                    ha='center', transform=ax.transAxes, color='blue')

    # ── Top-k / Top-p 비교 ──
    base_probs = softmax_with_temp(raw_logits, 1.0)

    # Top-k = 3
    ax = axes[1, 0]
    topk_probs = top_k_filter(base_probs, k=3)
    colors = ['#E74C3C' if p > 0 else '#D5D5D5' for p in topk_probs]
    ax.bar(range(len(topk_probs)), topk_probs, color=colors)
    ax.set_xticks(range(len(topk_probs)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("확률")
    ax.set_title("Top-k 샘플링 (k=3)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.text(0.5, 0.95, "상위 3개만 후보로 유지, 나머지 제거",
            fontsize=9, ha='center', transform=ax.transAxes, color='blue')

    # Top-p = 0.8
    ax = axes[1, 1]
    topp_probs = top_p_filter(base_probs, p=0.8)
    colors = ['#3498DB' if p > 0 else '#D5D5D5' for p in topp_probs]
    ax.bar(range(len(topp_probs)), topp_probs, color=colors)
    ax.set_xticks(range(len(topp_probs)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("확률")
    ax.set_title("Top-p 샘플링 (p=0.8)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.8)
    ax.text(0.5, 0.95, "누적 확률 80%가 될 때까지만 후보 유지",
            fontsize=9, ha='center', transform=ax.transAxes, color='blue')

    # 비교 설명
    ax = axes[1, 2]
    ax.axis('off')
    explanation = (
        "샘플링 전략 요약\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🔹 Temperature\n"
        "  T < 1: 높은 확률 토큰에 집중 (확정적)\n"
        "  T = 1: 원래 분포 유지\n"
        "  T > 1: 분포를 평탄하게 (창의적)\n\n"
        "🔹 Top-k\n"
        "  상위 k개 토큰만 후보로 유지\n"
        "  간단하지만 고정된 k가 항상 적절하지 않음\n\n"
        "🔹 Top-p (Nucleus Sampling)\n"
        "  누적 확률이 p가 될 때까지만 후보 유지\n"
        "  분포에 따라 후보 수가 동적으로 변함\n\n"
        "💡 실전: Temperature + Top-p 조합이 일반적"
    )
    ax.text(0.05, 0.95, explanation, fontsize=11, va='top',
            transform=ax.transAxes, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='orange'),
            fontfamily='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "04_sampling_strategies.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part C: 자기회귀(Autoregressive) 생성 과정 시각화
# =============================================================================
def visualize_autoregressive_generation():
    """
    LLM이 토큰을 하나씩 순차적으로 생성하는 자기회귀 과정을 시각화
    각 단계에서 이전에 생성된 모든 토큰이 다음 토큰 예측의 입력이 됨
    """
    from transformers import GPT2LMHeadModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # 단계별 생성 기록
    generation_steps = []
    current_ids = input_ids.clone()

    for step in range(6):  # 6개 토큰 생성
        with torch.no_grad():
            outputs = model(current_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # 상위 5개 후보
        top5_probs, top5_idx = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([idx]).strip() for idx in top5_idx]

        # Greedy 선택 (최고 확률 토큰)
        next_token_id = top5_idx[0].unsqueeze(0).unsqueeze(0)
        chosen_token = top5_tokens[0]

        generation_steps.append({
            'input': tokenizer.decode(current_ids[0]),
            'candidates': list(zip(top5_tokens, top5_probs.numpy())),
            'chosen': chosen_token,
        })

        current_ids = torch.cat([current_ids, next_token_id], dim=-1)

    # ── 시각화 ──
    fig, axes = plt.subplots(len(generation_steps), 1,
                             figsize=(18, 3.5 * len(generation_steps)))
    fig.suptitle("자기회귀(Autoregressive) 텍스트 생성 과정\n"
                 "— 각 단계에서 다음 토큰을 예측하고 선택 —",
                 fontsize=16, fontweight='bold', y=0.99)

    for idx, step in enumerate(generation_steps):
        ax = axes[idx]
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 현재 입력 시퀀스
        ax.text(0.01, 0.85, f"Step {idx+1} 입력:",
                fontsize=10, color='gray', transform=ax.transAxes)
        ax.text(0.13, 0.85, step['input'],
                fontsize=12, fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F0FE',
                          edgecolor='#4A90D9'))

        # 후보 토큰들과 확률
        ax.text(0.01, 0.45, "후보:", fontsize=10, color='gray',
                transform=ax.transAxes)
        x = 0.10
        for i, (tok, prob) in enumerate(step['candidates']):
            is_chosen = (i == 0)
            bg = '#2ECC71' if is_chosen else '#F5F5F5'
            ec = '#27AE60' if is_chosen else '#CCCCCC'
            lw = 3 if is_chosen else 1

            rect = mpatches.FancyBboxPatch(
                (x, 0.25), 0.14, 0.35,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor=ec, linewidth=lw,
                alpha=0.7 if is_chosen else 0.4,
                transform=ax.transAxes)
            ax.add_patch(rect)

            ax.text(x + 0.07, 0.48, f'"{tok}"',
                    fontsize=11, ha='center', va='center',
                    fontweight='bold' if is_chosen else 'normal',
                    transform=ax.transAxes)
            ax.text(x + 0.07, 0.32, f"{prob:.1%}",
                    fontsize=10, ha='center', va='center',
                    color='darkgreen' if is_chosen else 'gray',
                    transform=ax.transAxes)

            if is_chosen:
                ax.text(x + 0.07, 0.15, "← 선택",
                        fontsize=9, ha='center', color='green',
                        fontweight='bold', transform=ax.transAxes)

            x += 0.16

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "04_autoregressive_generation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()

    # 최종 생성 결과 출력
    final_text = tokenizer.decode(current_ids[0])
    print(f"\n  최종 생성 텍스트: \"{final_text}\"")


if __name__ == "__main__":
    print("=" * 60)
    print("  4. 텍스트 생성(Text Generation) 과정 시각화")
    print("=" * 60)
    print("\n[A] 다음 토큰 예측 확률 분포...")
    visualize_next_token_prediction()
    print("\n[B] 샘플링 전략 비교 (Temperature/Top-k/Top-p)...")
    visualize_sampling_strategies()
    print("\n[C] 자기회귀 생성 과정...")
    visualize_autoregressive_generation()
    print(f"\n완료! 이미지: {OUTPUT_DIR}")
