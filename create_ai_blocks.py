import urllib.request, json
blocks = [
    {"name": "AI Data Cleaner", "code": "import pandas as pd\nimport numpy as np\ndef clean(df):\n    df = df.drop_duplicates()\n    for col in df.select_dtypes(include=[np.number]).columns:\n        df[col] = df[col].fillna(df[col].median())\n        q1, q3 = df[col].quantile([0.25, 0.75])\n        iqr = q3 - q1\n        df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]\n    return df", "icon": "🧹", "category": "ai"},
    {"name": "AI Feature Engineer", "code": "from sklearn.preprocessing import PolynomialFeatures\ndef engineer(df, degree=2):\n    poly = PolynomialFeatures(degree=degree, include_bias=False)\n    return pd.DataFrame(poly.fit_transform(df.select_dtypes(include=[np.number])), columns=poly.get_feature_names_out())", "icon": "⚙️", "category": "ai"},
    {"name": "AI Model Selector", "code": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom xgboost import XGBClassifier\nfrom sklearn.metrics import accuracy_score\nmodels = {'LR': LogisticRegression(max_iter=1000), 'RF': RandomForestClassifier(), 'GB': GradientBoostingClassifier(), 'XGB': XGBClassifier()}\nbest = max(models.items(), key=lambda x: accuracy_score(y_test, x[1].fit(X_train, y_train).predict(X_test)))\nprint(f'Best: {best[0]}')", "icon": "🤖", "category": "ai"},
    {"name": "AI HyperTuner", "code": "from sklearn.model_selection import GridSearchCV\ndef tune(model, param_grid, X, y):\n    search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')\n    search.fit(X, y)\n    return search.best_estimator_, search.best_params_", "icon": "🎛️", "category": "ai"},
    {"name": "AI Visualizer", "code": "import matplotlib.pyplot as plt\ndef visualize(df, target):\n    fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n    df[target].value_counts().plot(kind='bar', ax=axes[0,0])\n    plt.tight_layout()\n    plt.savefig('analysis.png')\n    return fig", "icon": "📈", "category": "ai"},
    {"name": "AI Report Generator", "code": "import json\nfrom datetime import datetime\ndef report(model_name, metrics):\n    r = {'timestamp': datetime.now().isoformat(), 'model': model_name, 'metrics': metrics}\n    with open('ml_report.json', 'w') as f:\n        json.dump(r, f, indent=2)\n    return r", "icon": "📋", "category": "ai"},
]
for b in blocks:
    d = json.dumps(b).encode()
    r = urllib.request.Request("http://localhost:8000/api/blocks/create", data=d, headers={"Content-Type": "application/json"})
    print(json.loads(urllib.request.urlopen(r).read())["name"])
print("Done!")
