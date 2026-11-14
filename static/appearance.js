/* Локальный режим оформления. Не затрагивает профиль и настройки других устройств. */
window.GenesisAppearance = (() => {
  const storageKey = 'genesis_glass_beta';
  let enabled = true;
  try { enabled = localStorage.getItem(storageKey) !== 'off'; } catch {}
  function apply() {
    document.documentElement.dataset.glass = enabled ? 'beta' : 'off';
    document.querySelectorAll('[data-glass-toggle]').forEach(button => {
      button.setAttribute('aria-pressed',String(enabled));
      button.setAttribute('aria-label','Liquid Glass beta');
    });
  }
  function toggle() {
    enabled = !enabled; apply();
    let saved = true;
    try { localStorage.setItem(storageKey,enabled ? 'on' : 'off'); } catch { saved = false; }
    const status = document.getElementById('appearance-status');
    if (status) status.textContent = 'Liquid Glass beta ' + (enabled ? 'on.' : 'off.') + (saved ? '' : ' Browser storage is unavailable; this choice lasts until reload.');
  }
  apply();
  document.addEventListener('DOMContentLoaded',apply);
  document.addEventListener('click',event => { if (event.target.closest('[data-glass-toggle]')) toggle(); });
  window.addEventListener('storage',event => {
    if (event.key !== storageKey && event.key !== null) return;
    enabled = event.newValue !== 'off'; apply();
  });
  return {toggle};
})();
