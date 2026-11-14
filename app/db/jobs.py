"""Постоянная очередь с захватом задания и ограниченными повторами."""
import json
import time
import uuid


class JobQueueMixin:
    def queue_handoff(self,user_id,agent_id,conversation_id,user_message,reason,url,body):
        with self.write_transaction() as conn:
            row = conn.execute("INSERT INTO handoff_events(agent_id,conversation_id,user_message,reason,delivered,delivery_error) "
                               "VALUES (?,?,?,?,0,'Queued for delivery') RETURNING id",(agent_id,conversation_id,user_message,reason)).fetchone()
            payload = {"url":url,"body":dict(body,event_id=f"handoff:{row[0]}"),"event_id":row[0]}
            return self.enqueue_outgoing(user_id,"handoff",payload,event_id=f"handoff:{row[0]}",conn=conn)

    def enqueue_job(self, user_id, kind, payload, job_id=None):
        job_id = job_id or uuid.uuid4().hex
        now = int(time.time())
        with self.write_transaction() as conn:
            if self.is_postgres:
                conn.execute("SELECT id FROM users WHERE id=? FOR UPDATE", (user_id,))
            existing = conn.execute("SELECT id,kind,payload_json FROM background_jobs WHERE id=? AND user_id=?", (job_id,user_id)).fetchone()
            if existing:
                if existing[1] != kind or json.loads(existing[2]) != payload:
                    raise ValueError("Idempotency key was used with a different request")
                return job_id
            count = conn.execute("SELECT COUNT(*) FROM background_jobs WHERE user_id=? AND status IN ('pending','running')", (user_id,)).fetchone()[0]
            if count >= 2:
                raise ValueError("At most two active jobs per account")
            conn.execute("INSERT INTO background_jobs(id,user_id,kind,payload_json,status,created_at,updated_at) VALUES (?,?,?,?,'pending',?,?)",
                         (job_id,user_id,kind,json.dumps(payload),now,now))
        return job_id

    def get_job(self, user_id, job_id):
        with self._connect() as conn:
            row = conn.execute("SELECT id,kind,status,result_json,error,attempts,created_at,updated_at,progress_json FROM background_jobs WHERE id=? AND user_id=?", (job_id,user_id)).fetchone()
        if not row:
            return None
        result = dict(zip(("id","kind","status","result","error","attempts","created_at","updated_at","progress"),row))
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["progress"] = json.loads(result["progress"]) if result["progress"] else None
        return result

    def list_jobs(self, user_id):
        with self._connect() as conn:
            rows = conn.execute("SELECT id,kind,status,attempts,created_at,updated_at FROM background_jobs "
                                "WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT 50", (user_id,)).fetchall()
        return [dict(zip(("id", "kind", "status", "attempts", "created_at", "updated_at"), row)) for row in rows]

    def set_job_progress(self,job_id,event,attempt=None):
        with self.write_transaction() as conn:
            conn.execute("UPDATE background_jobs SET progress_json=?,updated_at=? WHERE id=? AND status='running' AND (CAST(? AS INTEGER) IS NULL OR attempts=?)",
                         (json.dumps(event,default=str),int(time.time()),job_id,attempt,attempt))

    def cancel_job(self, user_id, job_id):
        with self.write_transaction() as conn:
            return bool(conn.execute("UPDATE background_jobs SET status='cancelled',updated_at=? WHERE id=? AND user_id=? AND status IN ('pending','running')",
                                     (int(time.time()),job_id,user_id)).rowcount)

    def claim_job(self):
        now = int(time.time())
        with self.write_transaction() as conn:
            conn.execute("UPDATE background_jobs SET status='failed',error='Worker lease expired',updated_at=? WHERE status='running' AND lease_until<? AND attempts>=3",(now,now))
            row = conn.execute("SELECT id,user_id,kind,payload_json,attempts FROM background_jobs WHERE "
                               "(status='pending' OR (status='running' AND lease_until<?)) AND attempts<3 ORDER BY created_at LIMIT 1", (now,)).fetchone()
            if not row:
                return None
            changed = conn.execute("UPDATE background_jobs SET status='running',progress_json=NULL,lease_until=?,attempts=attempts+1,updated_at=? "
                                   "WHERE id=? AND (status='pending' OR (status='running' AND lease_until<?))", (now+900,now,row[0],now)).rowcount
            return {"id":row[0],"user_id":row[1],"kind":row[2],"payload":json.loads(row[3]),"attempt":row[4]+1} if changed else None

    def release_job(self, job_id, attempt):
        with self.write_transaction() as conn:
            conn.execute("UPDATE background_jobs SET status=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,"
                         "error=CASE WHEN attempts<3 THEN NULL ELSE 'Worker retry limit reached' END,lease_until=0,updated_at=? "
                         "WHERE id=? AND status='running' AND attempts=?", (int(time.time()),job_id,attempt))

    def finish_job(self, job_id, result=None, error=None, attempt=None):
        with self.write_transaction() as conn:
            conn.execute("UPDATE background_jobs SET status=?,result_json=?,error=?,updated_at=?,lease_until=0 WHERE id=? AND status='running' AND (CAST(? AS INTEGER) IS NULL OR attempts=?)",
                         ("failed" if error else "succeeded", json.dumps(result,default=str),error,int(time.time()),job_id,attempt,attempt))

    def enqueue_outgoing(self, user_id, kind, payload, event_id=None, conn=None):
        event_id = event_id or uuid.uuid4().hex
        values = (event_id,user_id,kind,json.dumps(payload),int(time.time()))
        query = "INSERT INTO outgoing_events(id,user_id,kind,payload_json,status,next_attempt) VALUES (?,?,?,?,'pending',?) ON CONFLICT DO NOTHING"
        if conn:
            conn.execute(query,values)
        else:
            with self.write_transaction() as connection:
                connection.execute(query,values)
        return event_id

    def claim_outgoing(self):
        now = int(time.time())
        with self.write_transaction() as conn:
            conn.execute("UPDATE outgoing_events SET status='failed',error='Final delivery lease expired',lease_until=0 WHERE status='pending' AND attempts>=5 AND lease_until<?", (now,))
            row = conn.execute("SELECT id,user_id,kind,payload_json,attempts FROM outgoing_events WHERE status='pending' AND next_attempt<=? AND lease_until<? AND attempts<5 ORDER BY next_attempt LIMIT 1", (now,now)).fetchone()
            if not row:
                return None
            if not conn.execute("UPDATE outgoing_events SET lease_until=?,attempts=attempts+1 WHERE id=? AND lease_until<?", (now+60,row[0],now)).rowcount:
                return None
            return {"id":row[0],"user_id":row[1],"kind":row[2],"payload":json.loads(row[3]),"attempts":row[4]+1}

    def finish_outgoing(self, event, error=None):
        with self.write_transaction() as conn:
            status = "delivered" if error is None else ("failed" if event["attempts"] >= 5 else "pending")
            conn.execute("UPDATE outgoing_events SET status=?,error=?,lease_until=0,next_attempt=? WHERE id=? AND status='pending' AND attempts=?",
                         (status,error,int(time.time())+min(3600,2**event["attempts"]*10),event["id"],event["attempts"]))

    def record_incoming(self, event_id, user_id, payload):
        with self.write_transaction() as conn:
            return bool(conn.execute("INSERT INTO incoming_events(id,user_id,payload_json,status,created_at) VALUES (?,?,?,'pending',?) ON CONFLICT DO NOTHING",
                                     (event_id,user_id,json.dumps(payload),int(time.time()))).rowcount)

    def claim_incoming(self):
        now = int(time.time())
        with self.write_transaction() as conn:
            row = conn.execute("SELECT id,user_id,payload_json FROM incoming_events WHERE status='pending' AND lease_until<? ORDER BY created_at LIMIT 1", (now,)).fetchone()
            if not row:
                return None
            if not conn.execute("UPDATE incoming_events SET lease_until=? WHERE id=? AND status='pending' AND lease_until<?", (now+600,row[0],now)).rowcount:
                return None
            return {"id":row[0],"user_id":row[1],"payload":json.loads(row[2])}

    def complete_incoming(self, event, outgoing):
        with self.write_transaction() as conn:
            self.enqueue_outgoing(event["user_id"], "telegram", outgoing, event_id=event["id"]+":reply", conn=conn)
            conn.execute("UPDATE incoming_events SET status='processed',lease_until=0 WHERE id=?", (event["id"],))
