#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
2. 임베딩(Embedding) 시각화
=============================================================================
토큰이 벡터 공간에 매핑되는 과정과 단어 간 유사도를 시각화합니다.

- 임베딩 벡터가 무엇인지
- 의미적으로 유사한 단어가 벡터 공간에서 가까이 위치하는 것을 확인
- PCA/t-SNE를 이용한 2D 시각화

실행: conda activate agent && python 20260215_160921_02_embedding.py
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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
# Part A: GPT-2 임베딩 벡터 추출 및 유사도 히트맵
# =============================================================================
def visualize_embedding_similarity():
    """실제 GPT-2 임베딩에서 단어 간 코사인 유사도를 히트맵으로 시각화"""
    from transformers import GPT2Model, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # 의미적 관계를 볼 수 있는 단어 선택
    words = ["king", "queen", "man", "woman", "prince", "princess",
             "dog", "cat", "car", "bicycle", "happy", "sad"]

    # 각 단어의 임베딩 벡터 추출 (첫 번째 토큰 사용)
    embeddings = []
    valid_words = []
    with torch.no_grad():
        for word in words:
            ids = tokenizer.encode(word)
            # 임베딩 레이어에서 직접 추출
            emb = model.wte(torch.tensor([ids[0]])).squeeze().numpy()
            embeddings.append(emb)
            valid_words.append(word)

    embeddings = np.array(embeddings)

    # 코사인 유사도 행렬 계산
    sim_matrix = cosine_similarity(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("임베딩(Embedding) 시각화 — 단어 간 의미적 유사도",
                 fontsize=18, fontweight='bold')

    # 히트맵
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-0.3, vmax=0.5)
    ax.set_xticks(range(len(valid_words)))
    ax.set_yticks(range(len(valid_words)))
    ax.set_xticklabels(valid_words, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(valid_words, fontsize=10)
    ax.set_title("코사인 유사도 히트맵", fontsize=14, fontweight='bold')

    # 각 셀에 값 표시
    for i in range(len(valid_words)):
        for j in range(len(valid_words)):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}",
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(sim_matrix[i,j]) > 0.3 else 'black')

    fig.colorbar(im, ax=ax, shrink=0.8)

    # PCA 2D 시각화
    ax2 = axes[1]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    # 카테고리별 색상
    categories = {
        '왕족': ['king', 'queen', 'prince', 'princess'],
        '성별': ['man', 'woman'],
        '동물': ['dog', 'cat'],
        '탈것': ['car', 'bicycle'],
        '감정': ['happy', 'sad'],
    }
    colors = {'왕족': '#E74C3C', '성별': '#3498DB', '동물': '#2ECC71',
              '탈것': '#F39C12', '감정': '#9B59B6'}

    for cat, cat_words in categories.items():
        idxs = [valid_words.index(w) for w in cat_words if w in valid_words]
        ax2.scatter(coords[idxs, 0], coords[idxs, 1],
                    c=colors[cat], s=150, label=cat, zorder=5)
        for i in idxs:
            ax2.annotate(valid_words[i], (coords[i, 0], coords[i, 1]),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=11, fontweight='bold')

    ax2.set_title("PCA 2D 투영 — 의미 공간에서의 단어 위치", fontsize=14, fontweight='bold')
    ax2.set_xlabel(f"PC1 (분산 설명: {pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 (분산 설명: {pca.explained_variance_ratio_[1]:.1%})")
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_embedding_similarity.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part B: 임베딩이 의미를 어떻게 인코딩하는지 — 벡터 연산
# =============================================================================
def visualize_word_analogy():
    """
    king - man + woman ≈ queen 같은 벡터 연산을 시각화
    임베딩이 의미적 관계를 캡처하는 것을 보여줌
    """
    from transformers import GPT2Model, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    def get_embedding(word):
        ids = tokenizer.encode(word)
        with torch.no_grad():
            return model.wte(torch.tensor([ids[0]])).squeeze().numpy()

    # 벡터 연산 예시들
    analogies = [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "japan", "tokyo"),
        ("big", "bigger", "small", "smaller"),
    ]

    fig, axes = plt.subplots(len(analogies), 1, figsize=(16, 5 * len(analogies)))
    fig.suptitle("임베딩 벡터 연산 — 의미적 관계 인코딩",
                 fontsize=18, fontweight='bold', y=0.98)

    for idx, (a, b, c, expected) in enumerate(analogies):
        ax = axes[idx]

        # 임베딩 추출
        emb_a, emb_b, emb_c = get_embedding(a), get_embedding(b), get_embedding(c)
        emb_expected = get_embedding(expected)

        # 벡터 연산: a - b + c
        result_vec = emb_a - emb_b + emb_c

        # 유사도 계산
        sim_expected = cosine_similarity([result_vec], [emb_expected])[0][0]

        # PCA로 시각화
        all_vecs = np.array([emb_a, emb_b, emb_c, emb_expected, result_vec])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_vecs)

        labels = [a, b, c, expected, f"{a}-{b}+{c}"]
        point_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

        for i, (label, color) in enumerate(zip(labels, point_colors)):
            marker = '*' if i == 4 else 'o'
            size = 300 if i == 4 else 150
            ax.scatter(coords[i, 0], coords[i, 1],
                       c=color, s=size, marker=marker, zorder=5, edgecolors='black')
            ax.annotate(label, (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=12, fontweight='bold', color=color)

        # a→b와 c→expected 사이 화살표 (관계 표시)
        ax.annotate('', xy=coords[1], xytext=coords[0],
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
        ax.annotate('', xy=coords[3], xytext=coords[2],
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))

        ax.set_title(f'"{a}" - "{b}" + "{c}" ≈ "{expected}"  '
                     f'(코사인 유사도: {sim_expected:.3f})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "02_word_analogy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


# =============================================================================
# Part C: 임베딩 차원별 값 분포 시각화
# =============================================================================
def visualize_embedding_dimensions():
    """임베딩 벡터의 각 차원이 어떤 값을 가지는지 히트맵으로 시각화"""
    from transformers import GPT2Model, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    words = ["hello", "world", "king", "queen", "love", "hate"]
    embeddings = []
    with torch.no_grad():
        for w in words:
            ids = tokenizer.encode(w)
            emb = model.wte(torch.tensor([ids[0]])).squeeze().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    fig, ax = plt.subplots(figsize=(16, 6))

    # 처음 50개 차원만 표시 (768차원 전체는 너무 넓음)
    display_dims = 50
    im = ax.imshow(embeddings[:, :display_dims], cmap='RdBu_r',
                   aspect='auto', vmin=-0.5, vmax=0.5)

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=12, fontweight='bold')
    ax.set_xlabel(f"임베딩 차원 (처음 {display_dims}개 / 전체 768개)", fontsize=12)
    ax.set_title("단어별 임베딩 벡터 값 분포 (GPT-2, 768차원 중 50차원)",
                 fontsize=14, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8, label='값')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_embedding_dimensions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[저장됨] {path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  2. 임베딩(Embedding) 시각화")
    print("=" * 60)
    print("\n[A] 단어 간 유사도 & PCA 2D 시각화...")
    visualize_embedding_similarity()
    print("\n[B] 벡터 연산 (king - man + woman ≈ queen)...")
    visualize_word_analogy()
    print("\n[C] 임베딩 차원별 값 분포...")
    visualize_embedding_dimensions()
    print(f"\n완료! 이미지: {OUTPUT_DIR}")
