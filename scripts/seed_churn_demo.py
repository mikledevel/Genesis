"""Seed the "support bot + churn model" demo for ONE user.

What it does (all local, no LLM / no API keys needed):
  1. generates a realistic synthetic e-shop churn dataset  -> datasets/churn_demo_large.csv
  2. trains a model with the platform's own AutoML           (DatasetTrainer, model_type="auto")
  3. registers it in the ModelRegistry under your user id
  4. creates an UNPUBLISHED support bot linked to that model

Then open the app -> My bots -> the new bot -> Bot Settings: the "Linked ML model" field is
already set; publish the bot when you're happy.

    python scripts/seed_churn_demo.py --email you@example.com
    python scripts/seed_churn_demo.py --email you@example.com --rows 3000
    python scripts/seed_churn_demo.py --no-train          # only (re)write the CSV

Needs the project's requirements installed (scikit-learn, xgboost) for training; the CSV
generator itself needs only numpy + pandas.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FEATURES = ["age", "tenure_months", "orders_last_90d", "support_tickets_90d",
            "avg_order_value", "days_since_last_order"]
TARGET = "churn"

BOT_SPEC = {
    "name": "Бот поддержки с прогнозом оттока",
    "description": "Отвечает про доставку и возврат; оценивает риск ухода клиента по данным аккаунта.",
    "greeting": "Здравствуйте! Чем могу помочь по заказу, доставке или возврату?",
    "system_prompt": (
        "Ты вежливый бот поддержки интернет-магазина. Отвечай кратко, по делу, на русском. "
        "Если клиент недоволен, говорит что уйдёт, хочет отписаться или удалить аккаунт, "
        "спроси у него значения для оценки риска ухода и вызови predict_with_model. "
        "Результат 1 означает высокий риск ухода, 0 - низкий; учитывай уверенность модели. "
        "Никогда не называй клиенту сам прогноз или проценты - просто веди себя соответственно: "
        "при высоком риске предложи конкретную помощь (быстрый возврат, компенсацию) и "
        "передай диалог человеку через request_handoff."
    ),
}


def generate_churn_dataset(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Synthetic but plausible: customers who haven't ordered for a long time, order rarely,
    open many support tickets and are new are likelier to churn. Deterministic for a seed.
    All features are numeric on purpose - the bot's predict tool builds a numeric vector."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    tenure = rng.integers(1, 60, n)
    orders = rng.poisson(3, n)
    tickets = rng.poisson(1, n)
    aov = np.round(rng.gamma(4, 15, n) + 10, 2)
    days_since = np.minimum(rng.exponential(30, n).astype(int), 365)
    logit = (-1.6 + 0.03 * days_since - 0.22 * orders + 0.45 * tickets - 0.02 * tenure)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"age": age, "tenure_months": tenure, "orders_last_90d": orders,
                         "support_tickets_90d": tickets, "avg_order_value": aov,
                         "days_since_last_order": days_since, TARGET: churn})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", help="email of the account that should own the bot and model")
    ap.add_argument("--user-id", help="alternative to --email")
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--no-train", action="store_true", help="only write the CSV")
    args = ap.parse_args()

    os.chdir(ROOT)  # the platform stores data under ./datasets, ./models, ./registry, ./agents
    df = generate_churn_dataset(args.rows)
    csv_path = os.path.join("datasets", "churn_demo_large.csv")
    os.makedirs("datasets", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[1/4] wrote {csv_path}: {len(df)} rows, churn rate {df[TARGET].mean():.0%}")
    if args.no_train:
        return 0

    user_id = args.user_id
    if not user_id:
        if not args.email:
            ap.error("pass --email (or --user-id) so the bot has an owner")
        from app.db.database import GenesisDB
        user = GenesisDB().get_user_by_email(args.email)
        if not user:
            print(f"No account with email {args.email!r}. Register in the app first.")
            return 1
        user_id = user["id"]

    from app.core.engine.trainer import DatasetTrainer
    from app.core.engine.registry import ModelRegistry
    from app.core.bot_builder import BotBuilder
    from app.core.agent_store import AgentStore

    result = DatasetTrainer().train(df, TARGET, "binary_classification", model_type="auto")
    print(f"[2/4] trained: {result.model_type}, test accuracy {result.metrics.get('accuracy_test')}")
    if result.feature_columns != FEATURES:
        print(f"Unexpected feature columns {result.feature_columns}; aborting.")
        return 1

    with open(result.model_path, "rb") as f:
        model = pickle.load(f)   # our own file, written a second ago by DatasetTrainer above
    model_id = ModelRegistry().register(
        model=model, metrics={k: v for k, v in result.metrics.items() if isinstance(v, (int, float))},
        task="binary_classification", model_type=result.model_type,
        features=result.feature_columns, target=TARGET,
        metadata={"author": user_id, "built_via": "seed_churn_demo"})
    print(f"[3/4] registered model {model_id}")

    bot = BotBuilder(user_id=user_id).publish(BOT_SPEC, author=user_id, publish=False)
    AgentStore().update_agent(bot["id"], {"linked_model_id": model_id}, owner_user_id=user_id)
    print(f"[4/4] created unpublished bot {bot['id']} linked to the model.")
    print("Open the app -> My bots -> Bot Settings to review it and publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
