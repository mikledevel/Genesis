"""Проверка staging по HTTP. Пароль читается из приватного JSON, не выводится."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import uuid
import httpx


def check(base_url, credentials, checkpoint, verify_only=False):
    login = json.loads(credentials.read_text())
    state = json.loads(checkpoint.read_text()) if verify_only else {}
    with httpx.Client(base_url=base_url, timeout=60) as client:
        for route in ('/health', '/ready', '/'):
            assert client.get(route).status_code == 200, route
        response = client.post('/api/auth/login', json={'email': login['email'], 'password': login['password']})
        assert response.status_code == 200, 'Login failed'
        client.headers['Authorization'] = 'Bearer ' + response.json()['token']
        if not verify_only:
            rng = random.Random(42)
            rows = ['age,income,churn']
            for _ in range(160):
                age = rng.randrange(18, 70)
                rows.append(f'{age},{rng.randrange(100,1000)},{int((age > 40) ^ (rng.random() < .15))}')
            upload = client.post('/api/agent/upload', files={'file': ('churn.csv', '\n'.join(rows).encode(), 'text/csv')})
            assert upload.status_code == 200, 'Upload failed'
            data = {'dataset_id': upload.json()['dataset_id'], 'target_column': 'churn', 'model_type': 'logistic', 'budget': 'fast'}
            key = uuid.uuid4().hex
            first = client.post('/api/jobs', json={'kind': 'train', 'payload': data}, headers={'Idempotency-Key': key})
            assert first.status_code == 202, 'Queue failed'
            state['job_id'] = first.json()['job_id']
            again = client.post('/api/jobs', json={'kind': 'train', 'payload': data}, headers={'Idempotency-Key': key})
            assert again.status_code == 202 and again.json()['job_id'] == state['job_id'], 'Idempotency failed'
            for _ in range(120):
                job = client.get('/api/jobs/' + state['job_id']).json()
                if job['status'] in ('succeeded', 'failed', 'cancelled'):
                    break
                time.sleep(1)
            assert job['status'] == 'succeeded', 'Training did not succeed'
            state['model_id'] = job['result']['data']['model_id']
            created = client.post('/api/keys/create', json={'user_name': 'Staging smoke', 'calls_limit': 20})
            assert created.status_code == 200, 'API key creation failed'
            state['api_key'] = created.json()['api_key']
            cancelled = client.post('/api/jobs', json={'kind': 'train', 'payload': data})
            assert cancelled.status_code == 202, 'Cancellation job not queued'
            state['cancelled_job_id'] = cancelled.json()['job_id']
            assert client.delete('/api/jobs/' + state['cancelled_job_id']).status_code == 200, 'Cancel failed'
        assert client.get('/api/jobs/' + state['job_id']).json()['status'] == 'succeeded'
        assert client.get('/api/jobs/' + state['cancelled_job_id']).json()['status'] == 'cancelled'
        record = client.get('/api/registry/models/' + state['model_id'])
        assert record.status_code == 200
        features = [{'age': 35, 'income': 500}[name] for name in record.json()['features']]
        prediction = client.post('/api/models/' + state['model_id'] + '/predict',
                                 headers={'X-API-Key': state['api_key']}, json={'features': features})
        assert prediction.status_code == 200, 'Saved model prediction failed'
        value = prediction.json()['prediction']
        if verify_only:
            assert value == state['prediction'], 'Prediction changed after restart'
        else:
            state['prediction'] = value
        # Checkpoint содержит API-ключ: сохраняем его только локально с правами 0600.
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(checkpoint, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(checkpoint, 0o600)
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(state, handle, indent=2)
    print('Restart verification passed' if verify_only else 'HTTP staging smoke passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    check(args.base_url, args.credentials, args.checkpoint, args.verify_only)
