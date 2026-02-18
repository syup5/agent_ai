#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
1. 토큰화(Tokenization) 시각화
=============================================================================
LLM이 문자열을 인식하는 첫 번째 단계: 텍스트를 토큰으로 분리하는 과정을 시각화합니다.

- 입력 문자열 → 토큰 분리 → 토큰 ID 변환
- BPE(Byte Pair Encoding) 알고리즘 동작 원리
- 전체 파이프라인 시각화

실행: conda activate agent && python 20260215_160921_01_tokenization.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──
def setup_korean_font():
    import subprocess
    # Noto Sans CJK KR 폰트 경로를 시스템에서 직접 찾아 등록
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
    # fallback
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
# Part A: 실제 토크나이저를 사용한 토큰화 시각화
# =============================================================================
def visualize_tokenization():
    """GPT-2 BPE 토크나이저로 다양한 문장을 토큰화하여 시각화"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    sentences = [
        "Hello, how are you?",
        "The transformer model is powerful.",
        "인공지능은 미래를 바꿀 것입니다.",
        "unhappiness",
    ]

    fig, axes = plt.subplots(len(sentences), 1, figsize=(18, 3.5 * len(sentences)))
    fig.suptitle("토큰화(Tokenization) 시각화 — GPT-2 BPE 토크나이저",
                 fontsize=18, fontweight='bold', y=0.98)

    cmap = plt.cm.get_cmap('tab20')

    for idx, sentence in enumerate(sentences):
        ax = axes[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        encoded = tokenizer.encode(sentence)
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        decoded_tokens = [tokenizer.decode([tid]) for tid in encoded]

        # 원본 문장
        ax.text(0.01, 0.85, f'원본: "{sentence}"',
                fontsize=13, fontweight='bold', va='top', transform=ax.transAxes)

        # 토큰 박스
        n_tokens = len(tokens)
        box_width = min(0.12, 0.90 / n_tokens)
        start_x = 0.02

        for i, (token, tid, decoded) in enumerate(zip(tokens, encoded, decoded_tokens)):
            color = cmap(i % 20)
            x = start_x + i * (box_width + 0.005)
            if x + box_width > 0.98:
                break

            rect = mpatches.FancyBboxPatch(
                (x, 0.35), box_width, 0.40,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=0.3, edgecolor=color, linewidth=2,
                transform=ax.transAxes
            )
            ax.add_patch(rect)

            # 토큰 텍스트
            display = repr(decoded)[1:-1]
            ax.text(x + box_width/2, 0.60, display,
                    fontsize=9, ha='center', va='center',
                    fontweight='bold', transform=ax.transAxes)

            # 토큰 ID
            ax.text(x + box_width/2, 0.42, f"ID:{tid}",
                    fontsize=7, ha='center', va='center',
                    color='gray', transform=ax.transAxes)

        ax.text(0.99, 0.85, f"토큰 수: {n_tokens}",
                fontsize=11, ha='right', va='top', color='red',
                fontweight='bold', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "01_tokenization.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part B: BPE 알고리즘 단계별 시각화
# =============================================================================
def visualize_bpe():
    """BPE가 단계별로 토큰 어휘를 구축하는 과정을 시각화"""
    corpus = {
        'l o w </w>': 5,
        'l o w e r </w>': 2,
        'n e w e s t </w>': 6,
        'w i d e s t </w>': 3,
    }

    def get_pairs(corp):
        pairs = {}
        for word, freq in corp.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def merge_pair(pair, corp):
        new_corp = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in corp.items():
            new_corp[word.replace(bigram, replacement)] = freq
        return new_corp

    steps = [{'corpus': dict(corpus), 'merged': None, 'freq': None}]
    current = dict(corpus)
    for _ in range(6):
        pairs = get_pairs(current)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        current = merge_pair(best, current)
        steps.append({'corpus': dict(current), 'merged': best, 'freq': pairs[best]})

    n = min(len(steps), 5)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n))
    fig.suptitle("BPE(Byte Pair Encoding) 알고리즘 동작 과정",
                 fontsize=18, fontweight='bold', y=0.99)

    for si in range(n):
        ax = axes[si]
        ax.axis('off')
        step = steps[si]

        if si == 0:
            title = "초기 상태: 모든 문자를 개별 토큰으로 분리"
        else:
            m = step['merged']
            title = f"단계 {si}: '{m[0]}' + '{m[1]}' → '{m[0]}{m[1]}' (빈도: {step['freq']})"

        ax.text(0.01, 0.95, title, fontsize=13, fontweight='bold',
                va='top', color='darkblue', transform=ax.transAxes)

        y = 0.70
        for word, freq in step['corpus'].items():
            symbols = word.split()
            ax.text(0.03, y, f"빈도 {freq}:", fontsize=10, va='center',
                    color='gray', transform=ax.transAxes)
            x = 0.15
            for sym in symbols:
                is_new = (step['merged'] and len(sym) > 1
                          and sym != '</w>' and sym == ''.join(step['merged']))
                bg = '#FFD700' if is_new else '#E8F0FE'
                ec = '#FF4500' if is_new else '#4A90D9'
                rect = mpatches.FancyBboxPatch(
                    (x, y-0.06), 0.055, 0.12,
                    boxstyle="round,pad=0.01",
                    facecolor=bg, edgecolor=ec, linewidth=1.5,
                    transform=ax.transAxes
                )
                ax.add_patch(rect)
                ax.text(x+0.0275, y, sym, fontsize=8, ha='center', va='center',
                        fontweight='bold', transform=ax.transAxes)
                x += 0.06
            y -= 0.22

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, "01_bpe_process.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part C: 텍스트 → 토큰 → ID → 임베딩 벡터 파이프라인
# =============================================================================
def visualize_pipeline():
    """토큰화 전체 파이프라인(텍스트→토큰→ID→벡터)을 시각화"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "AI changes the world"

    encoded = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(encoded)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.97, "토큰화 파이프라인: 텍스트 → 토큰 → ID → 벡터",
            fontsize=18, fontweight='bold', ha='center', va='top')

    # 단계 1: 원본 텍스트
    ax.text(0.5, 0.88, f'입력: "{text}"', fontsize=15, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8E8',
                      edgecolor='green', linewidth=2))

    ax.annotate('', xy=(0.5, 0.78), xytext=(0.5, 0.83),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(0.65, 0.805, '① 토큰화 (BPE)', fontsize=11, color='gray')

    # 단계 2: 토큰
    n = len(tokens)
    bw = 0.11
    total = n * bw + (n-1) * 0.01
    sx = 0.5 - total / 2
    cmap = plt.cm.get_cmap('Pastel1')

    for i, tok in enumerate(tokens):
        x = sx + i * (bw + 0.01)
        rect = mpatches.FancyBboxPatch(
            (x, 0.72), bw, 0.06,
            boxstyle="round,pad=0.01",
            facecolor=cmap(i/max(n-1,1)), edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+bw/2, 0.75, tok.replace('Ġ', '_ '),
                fontsize=10, ha='center', va='center', fontweight='bold')

    ax.text(0.06, 0.75, '토큰:', fontsize=12, fontweight='bold', va='center')

    ax.annotate('', xy=(0.5, 0.63), xytext=(0.5, 0.70),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(0.65, 0.665, '② 어휘사전 ID 조회', fontsize=11, color='gray')

    # 단계 3: 토큰 ID
    for i, tid in enumerate(encoded):
        x = sx + i * (bw + 0.01)
        rect = mpatches.FancyBboxPatch(
            (x, 0.56), bw, 0.06,
            boxstyle="round,pad=0.01",
            facecolor='#FFF3CD', edgecolor='orange', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+bw/2, 0.59, str(tid),
                fontsize=10, ha='center', va='center', fontweight='bold')

    ax.text(0.06, 0.59, '토큰ID:', fontsize=12, fontweight='bold', va='center')

    ax.annotate('', xy=(0.5, 0.46), xytext=(0.5, 0.54),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(0.65, 0.50, '③ 임베딩 룩업', fontsize=11, color='gray')

    # 단계 4: 임베딩 벡터 (시뮬레이션)
    np.random.seed(42)
    embed_dim = 8

    for i in range(n):
        x = sx + i * (bw + 0.01)
        vec = np.random.randn(embed_dim) * 0.5
        rect = mpatches.FancyBboxPatch(
            (x, 0.18), bw, 0.27,
            boxstyle="round,pad=0.01",
            facecolor='#E8E0F0', edgecolor='purple', linewidth=1.5)
        ax.add_patch(rect)
        for j, val in enumerate(vec):
            y = 0.42 - j * 0.03
            c = plt.cm.RdBu(0.5 + val/2)
            ax.text(x+bw/2, y, f"{val:+.2f}",
                    fontsize=7, ha='center', va='center', color=c, fontweight='bold')

    ax.text(0.06, 0.32, f'임베딩\n(d={embed_dim}):', fontsize=11,
            fontweight='bold', va='center', ha='center')

    ax.text(0.5, 0.08,
            "실제 GPT-2: 어휘 크기=50,257 | 임베딩 차원=768\n"
            "각 토큰 ID → 768차원 밀집 벡터 → Transformer 입력",
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0',
                      edgecolor='gray'), linespacing=1.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_tokenization_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  1. 토큰화(Tokenization) 시각화")
    print("=" * 60)
    print("\n[A] 실제 토크나이저 토큰화...")
    visualize_tokenization()
    print("\n[B] BPE 알고리즘 동작 원리...")
    visualize_bpe()
    print("\n[C] 토큰화 파이프라인 전체 흐름...")
    visualize_pipeline()
    print(f"\n완료! 이미지: {OUTPUT_DIR}")
