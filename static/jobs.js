/* Постоянные задачи: повторное открытие диалога возобновляет проверку статуса. */
window.GenesisJobs = (() => {
  function attach(bubble, jobId) {
    const box = document.createElement('section'); box.className = 'job-status data-card';
    box.setAttribute('aria-label', 'Background job');
    const status = document.createElement('p'); status.setAttribute('role', 'status'); status.textContent = 'Loading job status…';
    const action = document.createElement('button'); action.type = 'button'; action.className = 'btn btn-secondary';
    action.textContent = 'Cancel job';
    box.append(status, action); bubble.append(box);
    let timer, stopped = false;
    const url = API + '/jobs/' + encodeURIComponent(jobId);
    async function poll() {
      if (!box.isConnected || stopped) return;
      clearTimeout(timer);
      try {
        const response = await apiFetch(url);
        if (!response.ok) throw new Error('Status unavailable');
        const job = await response.json();
        if (!box.isConnected || stopped) return;
        const terminal = ['succeeded', 'failed', 'cancelled'].includes(job.status);
        const detail = terminal ? '' : job.progress?.detail || job.progress?.title || job.progress?.label;
        status.textContent = 'Job ' + job.status + (detail ? ' · ' + detail : '');
        if (['succeeded', 'failed', 'cancelled'].includes(job.status)) {
          stopped = true; action.remove();
          const result = job.result;
          if (result && (result.message || result.data || result.code || result.metrics)) {
            const output = document.createElement('div'); output.className = 'msg-col';
            const answer = document.createElement('div'); answer.className = 'bubble'; output.append(answer);
            const wrapper = document.createElement('div'); wrapper.append(output); box.append(wrapper);
            const trained = job.status === 'succeeded' && ['train','model_create'].includes(job.kind) && result.data?.model_id && result.data?.metrics;
            const message = trained ? 'The model is trained and saved in your registry.' : result.message || (job.status === 'succeeded' ? 'Job completed.' : 'The job needs your attention.');
            fillAiMsg(wrapper, message, result.data || result, { replay: true });
            output.querySelector('.msg-actions')?.remove();
          } else if (job.error) {
            const detail = document.createElement('p'); detail.textContent = job.error; box.append(detail);
          }
          return;
        }
        action.textContent = 'Cancel job'; action.disabled = false; action.onclick = cancel;
        timer = setTimeout(poll, 2000);
      } catch {
        status.textContent = 'Could not load job status. The job may still be running.';
        action.disabled = false; action.textContent = 'Retry status'; action.onclick = poll;
      }
    }
    async function cancel() {
      action.disabled = true; clearTimeout(timer);
      try {
        const response = await apiFetch(url, { method: 'DELETE' });
        if (!response.ok && response.status !== 409) throw new Error('Cancellation unavailable');
        await poll();
      } catch {
        status.textContent = 'Could not cancel the job. Check its status before retrying.';
        action.disabled = false; action.textContent = 'Retry status'; action.onclick = poll;
      }
    }
    action.onclick = cancel;
    poll();
  }
  return { attach };
})();
