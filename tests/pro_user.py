"""Test helper: server-side training is a Pro feature (app/core/plans.py, "server_training"). Tests that exercise the
training code itself - not the plans - use a real Pro account, exactly like a paying customer. An unknown user id is
treated as Free, so the id has to exist in the database the agent under test uses."""


def make_pro_user(db, user_id: str, plan: str = "pro") -> str:
    with db._connect() as conn:
        conn.execute("INSERT OR IGNORE INTO users (id, email, password_hash, name, email_canonical) VALUES (?, ?, 'x', 'T', ?)",
                     (user_id, f"{user_id}@pro.test", f"{user_id}@pro.test"))
        conn.commit()
    db.set_user_plan(user_id, plan)
    return user_id
