from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import random
import math

app = Flask(__name__)
DB_NAME = "japan_geography_attention.db"

# ==========================================
# 既存のロジック (DB構築 & Attentionクラス)
# ==========================================
# ※ 元のコードと同じ関数を使用しますが、print文などは減らしています

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS quizzes")
    cursor.execute("""
    CREATE TABLE quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT, question TEXT, options TEXT, correct_answer TEXT, explanation TEXT
    );
    """)
    conn.commit()
    conn.close()

def insert_sample_data():
    # データリストは長いので省略しませんが、元のコードと同じものを入れます
    quiz_data_list = [
        {"category": "地理", "question": "日本で最も面積が小さい都道府県はどこですか？", "options": ["大阪府", "香川県", "東京都", "沖縄県"], "correct_answer": "香川県", "explanation": "かつては大阪府が最小でしたが、現在は香川県です。"},
        {"category": "地理", "question": "日本で最も長い川はどれですか？", "options": ["利根川", "信濃川", "石狩川", "北上川"], "correct_answer": "信濃川", "explanation": "信濃川は全長367kmで日本一です。"},
        {"category": "地理", "question": "「海に面していない（内陸県）」は日本にいくつありますか？", "options": ["6つ", "8つ", "10つ", "12つ"], "correct_answer": "8つ", "explanation": "栃木、群馬、埼玉、山梨、長野、岐阜、滋賀、奈良の8県です。"},
        {"category": "地理", "question": "日本で最も深い湖はどこですか？", "options": ["琵琶湖", "摩周湖", "田沢湖", "支笏湖"], "correct_answer": "田沢湖", "explanation": "秋田県の田沢湖で水深423.4mです。"},
        {"category": "地理", "question": "日本の最南端にある有人島はどこですか？", "options": ["沖ノ鳥島", "与那国島", "波照間島", "南大東島"], "correct_answer": "波照間島", "explanation": "有人島の最南端は波照間島です。"},
        {"category": "地理", "question": "隣接する都道府県の数が最も多い県はどこですか？", "options": ["長野県", "埼玉県", "岐阜県", "京都府"], "correct_answer": "長野県", "explanation": "長野県は8つの県と隣接しています。"},
        {"category": "地理", "question": "鳥取砂丘がある鳥取県は、人口の多さでは全国何位ですか？", "options": ["最下位", "下から5番目", "真ん中", "上位"], "correct_answer": "最下位", "explanation": "鳥取県は日本で最も人口が少ないです。"},
        {"category": "地理", "question": "本州と四国の間にかかる橋のルートとして存在しないものは？", "options": ["神戸・鳴門", "児島・坂出", "尾道・今治", "和歌山・徳島"], "correct_answer": "和歌山・徳島", "explanation": "本州四国連絡橋は3ルートのみです。"},
        {"category": "地理", "question": "東京23区の中で最も面積が広い区はどこですか？", "options": ["世田谷区", "足立区", "練馬区", "大田区"], "correct_answer": "大田区", "explanation": "大田区が最大です。羽田空港が含まれます。"},
        {"category": "地理", "question": "九州地方に含まれない県はどれですか？", "options": ["佐賀県", "大分県", "山口県", "宮崎県"], "correct_answer": "山口県", "explanation": "山口県は中国地方です。"}
    ]
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    for q in quiz_data_list:
        cursor.execute("INSERT INTO quizzes (category, question, options, correct_answer, explanation) VALUES (?, ?, ?, ?, ?)",
                       (q['category'], q['question'], json.dumps(q['options'], ensure_ascii=False), q['correct_answer'], q['explanation']))
    conn.commit()
    conn.close()

class SimpleAttentionEngine:
    def __init__(self, texts):
        self.vocab = sorted(list(set("".join(texts))))
        self.d_k = len(self.vocab)

    def text_to_vector(self, text):
        vec = [0] * self.d_k
        for char in text:
            if char in self.vocab:
                idx = self.vocab.index(char)
                vec[idx] = 1
        return vec

    def dot_product(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]

    def compute_attention(self, query_text, key_texts):
        Q = self.text_to_vector(query_text)
        raw_scores = []
        for k_text in key_texts:
            K = self.text_to_vector(k_text)
            dot = self.dot_product(Q, K)
            scaled_score = dot / math.sqrt(self.d_k)
            raw_scores.append(scaled_score)
        return self.softmax(raw_scores)

# ==========================================
# Webアプリ用設定 (初期化)
# ==========================================
setup_database()
insert_sample_data()

# エンジンをグローバル変数として準備
conn = sqlite3.connect(DB_NAME)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT * FROM quizzes")
ROWS = cursor.fetchall() # 全データをメモリに保持
conn.close()

QUESTIONS_TEXT = [row['question'] for row in ROWS]
ENGINE = SimpleAttentionEngine(QUESTIONS_TEXT)

# ==========================================
# Flask ルーティング (API)
# ==========================================

@app.route('/')
def index():
    """トップページを表示"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_quiz():
    """フロントエンドから検索クエリを受け取り、Attentionで問題を返す"""
    data = request.json
    user_query = data.get('query', '')
    
    # Attention計算
    weights = ENGINE.compute_attention(user_query, QUESTIONS_TEXT)
    
    # 結果を結合
    results = []
    for i, weight in enumerate(weights):
        results.append({
            "index": i,
            "weight": weight,
            "question": QUESTIONS_TEXT[i]
        })
    
    # スコア順にソート
    results.sort(key=lambda x: x['weight'], reverse=True)
    top_result = results[0]
    
    # 閾値判定
    if top_result['weight'] < 0.15:
        target_row = random.choice(ROWS)
        message = "関連する問題が見つかりませんでした。ランダムに出題します。"
    else:
        target_row = ROWS[top_result['index']]
        message = f"Attentionスコア: {top_result['weight']:.4f}"

    # JSONで返すデータを作成
    response_data = {
        "message": message,
        "category": target_row['category'],
        "question": target_row['question'],
        "options": json.loads(target_row['options']),
        "correct_answer": target_row['correct_answer'],
        "explanation": target_row['explanation']
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)