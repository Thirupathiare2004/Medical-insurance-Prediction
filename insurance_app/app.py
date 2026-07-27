from flask import Flask, request, jsonify, render_template, session
import pandas as pd, numpy as np, joblib, json, os, datetime, uuid

app = Flask(__name__)
app.secret_key = 'insureai-secret-2024'

BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, 'models')
DATA   = os.path.join(BASE, 'datasets')

# ── load all models once ──────────────────────────────────────────────────────
def load(name):
    model    = joblib.load(f'{MODELS}/{name}_model.pkl')
    features = joblib.load(f'{MODELS}/{name}_features.pkl')
    with open(f'{MODELS}/{name}_encoders.json') as f:
        enc = json.load(f)
    with open(f'{MODELS}/{name}_metrics.json') as f:
        metrics = json.load(f)
    return model, features, enc, metrics

CACHE = {t: load(t) for t in ['medical','car','home','life','travel']}

LABELS = {
    'medical':'Medical','car':'Car','home':'Home','life':'Life','travel':'Travel'
}
ICONS = {
    'medical':'🏥','car':'🚗','home':'🏠','life':'💙','travel':'✈️'
}

# ── in-memory prediction history (per session) ────────────────────────────────
HISTORY = {}   # session_id -> list of records

def get_sid():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']

def encode_input(raw, features, enc):
    row = {}
    for feat in features:
        val = raw.get(feat, 0)
        if feat in enc:
            rev = {v: int(k) for k, v in enc[feat].items()}
            val = rev.get(str(val), 0)
        else:
            try: val = float(val)
            except: val = 0
        row[feat] = val
    return pd.DataFrame([row])

# ── routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/api/predict/<ins_type>', methods=['POST'])
def predict(ins_type):
    if ins_type not in CACHE:
        return jsonify({'error': 'Unknown insurance type'}), 400
    model, features, enc, metrics = CACHE[ins_type]
    data = request.json or {}
    try:
        X   = encode_input(data, features, enc)
        pred = float(model.predict(X)[0])
        imp = sorted(zip(features, model.feature_importances_), key=lambda x:-x[1])[:5]

        # percentile vs dataset
        df  = pd.read_csv(f'{DATA}/{ins_type}_insurance.csv')
        tgt = 'charges' if ins_type=='medical' else 'premium'
        pct = float(np.mean(df[tgt] <= pred) * 100)

        result = {
            'premium': round(pred, 2),
            'metrics': metrics,
            'percentile': round(pct, 1),
            'top_factors': [{'feature': f.replace('_',' ').title(),
                             'importance': round(v*100,1)} for f,v in imp]
        }

        # save history
        sid = get_sid()
        HISTORY.setdefault(sid, []).append({
            'id': str(uuid.uuid4())[:8],
            'type': ins_type,
            'label': LABELS[ins_type],
            'icon': ICONS[ins_type],
            'premium': round(pred,2),
            'time': datetime.datetime.now().strftime('%d %b %Y, %H:%M'),
            'inputs': data
        })

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics/<ins_type>')
def get_metrics(ins_type):
    if ins_type not in CACHE:
        return jsonify({'error': 'Unknown'}), 400
    return jsonify(CACHE[ins_type][3])

@app.route('/api/all_metrics')
def all_metrics():
    return jsonify({t: CACHE[t][3] for t in CACHE})

@app.route('/api/history')
def history():
    sid = get_sid()
    return jsonify(HISTORY.get(sid, []))

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    sid = get_sid()
    HISTORY[sid] = []
    return jsonify({'ok': True})

@app.route('/api/dataset_stats/<ins_type>')
def dataset_stats(ins_type):
    if ins_type not in CACHE:
        return jsonify({'error': 'Unknown'}), 400
    df  = pd.read_csv(f'{DATA}/{ins_type}_insurance.csv')
    tgt = 'charges' if ins_type=='medical' else 'premium'
    vals = df[tgt].tolist()
    # histogram buckets (10 bins)
    counts, edges = np.histogram(vals, bins=10)
    return jsonify({
        'mean':   round(float(df[tgt].mean()),2),
        'median': round(float(df[tgt].median()),2),
        'min':    round(float(df[tgt].min()),2),
        'max':    round(float(df[tgt].max()),2),
        'std':    round(float(df[tgt].std()),2),
        'histogram': {
            'counts': counts.tolist(),
            'edges':  [round(e,0) for e in edges.tolist()]
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
