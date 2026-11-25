"""Проверка разового отчёта на синтетике; без вызовов платёжного провайдера."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import httpx


def check(base_url, credentials, checkpoint, verify_only=False):
    login = json.loads(credentials.read_text())
    state = json.loads(checkpoint.read_text()) if verify_only else {}
    with httpx.Client(base_url=base_url.rstrip('/'), timeout=55) as client:
        assert client.get('/ready').status_code == 200, 'Service is not ready'
        response = client.post('/api/auth/login', json={'email':login['email'],'password':login['password']})
        assert response.status_code == 200, 'Login failed'
        client.headers['Authorization'] = 'Bearer '+response.json()['token']
        if not verify_only:
            from datetime import date, timedelta
            rng=random.Random(2451)
            rows=['date,sales']
            for index in range(240):
                value=100+0.1*index+15*math.sin(2*math.pi*index/7)+rng.gauss(0,2)
                rows.append(f'{date(2025,1,1)+timedelta(days=index)},{value:.4f}')
            upload=client.post('/api/agent/upload',files={'file':('synthetic-report.csv','\n'.join(rows).encode(),'text/csv')})
            assert upload.status_code==200, 'Synthetic upload failed'
            payload={'dataset_id':upload.json()['dataset_id'],'date_column':'date','target_column':'sales','horizon':14,'agg':'sum'}
            preview=client.post('/api/forecasts/preview',json=payload)
            assert preview.status_code==200 and preview.json()['eligible'] and preview.json()['can_run'], 'Preview failed'
            headers={'Idempotency-Key':uuid.uuid4().hex}
            response=client.post('/api/jobs',json={'kind':'forecast','payload':payload},headers=headers)
            assert response.status_code==202, 'Forecast did not queue'
            state['job_id']=response.json()['job_id']
            repeated=client.post('/api/jobs',json={'kind':'forecast','payload':payload},headers=headers)
            assert repeated.status_code==202 and repeated.json()['job_id']==state['job_id'], 'Duplicate forecast queued'
        for _ in range(120):
            response=client.get('/api/jobs/'+state['job_id'])
            assert response.status_code==200, 'Saved job missing'
            job=response.json()
            if job['status'] in ('succeeded','failed','cancelled'):
                break
            time.sleep(1)
        assert job['status']=='succeeded', 'Forecast failed or did not finish'
        future=job['result']['data']['report_snapshot']['forecast']['future']
        digests={}
        for language,output_format in [('ru','html'),('en','html'),('en','csv')]:
            route=f"/api/jobs/{state['job_id']}/report?lang={language}&format={output_format}"
            report=client.get(route);repeated=client.get(route)
            assert report.status_code==200 and report.content==repeated.content, 'Report missing or changed on repeat'
            assert report.headers.get('cache-control')=='no-store' and 'attachment' in report.headers.get('content-disposition',''), 'Unsafe download headers'
            if output_format=='csv':
                exported=list(csv.DictReader(io.StringIO(report.content.decode('utf-8-sig'))))
                assert len(exported)==14
                for index,row in enumerate(exported):
                    for column,key in [('forecast','values'),('lower','lower'),('upper','upper')]:
                        assert float(row[column])==future[key][index], 'Export changed saved numbers'
            else:
                assert '<script' not in report.text.lower(), 'Report must be autonomous'
            digests[language+'-'+output_format]=hashlib.sha256(report.content).hexdigest()
        if verify_only:
            assert digests==state['digests'], 'Saved reports changed after restart'
        else:
            state['digests']=digests
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            descriptor=os.open(checkpoint,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(descriptor,'w') as file:json.dump(state,file,indent=2)
        print('Saved RU/EN/CSV reports verified' + (' after restart' if verify_only else '; forecast queued exactly once'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--credentials',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    check(args.base_url,args.credentials,args.checkpoint,args.verify_only)
