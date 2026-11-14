/* Небольшой набор пиктограмм для данных и этапов пайплайна. */
window.GenesisIcons = (() => {
  const paths = {
    data:'M4 4h16v16H4z M4 9h16 M9 4v16 M9 14h11',
    transform:'M4 7h16 M4 17h16 M8 4v6 M16 14v6 M4 12h16',
    model:'M5 5h5v5H5z M14 14h5v5h-5z M14 5h5v5h-5z M7 10v7h7 M10 7h4 M16 10v4',
    evaluate:'M4 19V5 M4 19h16 M8 15l4-6 4 3 4-7',
    save:'M5 4h12l3 3v13H4V4h1 M8 4v6h8V4 M8 20v-6h8v6',
    folder:'M3 7V5h7l2 3h9v12H3z'
  };
  /** @param {string} name @returns {string} */
  function icon(name) {
    const path = paths[name] || paths.model;
    return '<svg class="product-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="square" stroke-linejoin="miter" aria-hidden="true"><path d="' + path + '"></path></svg>';
  }
  /** @param {string} id @returns {string} */
  function block(id) {
    const kind = id.startsWith('load') ? 'data' : id.startsWith('train') ? 'model' : /evaluate|validate|matrix|profile/.test(id) ? 'evaluate' : id === 'save' ? 'save' : 'transform';
    return icon(kind);
  }
  return {icon,block};
})();
