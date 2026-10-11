// Native menu selectors match the Streamlit version pinned in app/requirements.txt.
// Use native actions so widget colors and browser persistence stay in sync.
export default function({ parentElement, setStateValue, data }) {
  window.__floodlabThemeCleanup?.();
  const source = parentElement.querySelector('.theme-switcher');
  const menu = document.querySelector('[data-testid="stMainMenu"]');
  if (!menu) return;

  const host = document.createElement('div');
  host.dataset.floodlabThemeControl = 'true';
  const root = host.attachShadow({mode: 'open'});
  for (const style of parentElement.querySelectorAll('style')) {
    root.append(style.cloneNode(true));
  }
  const control = source.cloneNode(true);
  control.style.removeProperty('display');
  root.append(control);
  source.style.display = 'none';
  menu.before(host);

  const button = control.querySelector('button');
  const status = control.querySelector('.theme-status');
  const nativeMenu = () => document.querySelector('[data-testid="stMainMenuButton"]');
  const nativeChoice = mode => document.querySelector(`[data-testid="stMainMenuItem-theme-${mode}"]`);
  const cacheKey = window.__floodlabThemeCacheKey ||= `stActiveTheme-${location.pathname}-v2`;
  let busy = false;
  let disposed = false;
  let timer;
  let focusTimer;
  let reportedAppearance = data?.appearance;

  function reportAppearance(appearance) {
    if (appearance !== reportedAppearance) {
      reportedAppearance = appearance;
      setStateValue('appearance', appearance);
    }
  }

  function updateIcon(appearance) {
    control.dataset.appearance = appearance;
    const label = `Switch to ${appearance === 'dark' ? 'light' : 'dark'} theme`;
    button.setAttribute('aria-label', label);
    button.setAttribute('aria-pressed', String(appearance === 'dark'));
    button.title = label;
  }

  function restoreFocus() {
    clearTimeout(focusTimer);
    focusTimer = setTimeout(() => {
      if (!disposed && window.__floodlabThemeFocus) {
        button.focus({preventScroll: true});
        delete window.__floodlabThemeFocus;
      }
    }, 100);
  }

  function sync() {
    // Cloud adds GitHub and Share controls. Flow layout leaves each its own space.
    const menu = document.querySelector('[data-testid="stMainMenu"]');
    if (menu && host.nextSibling !== menu) menu.before(host);
    const styles = getComputedStyle(source);
    for (const token of ['--st-text-color', '--st-primary-color', '--st-background-color']) {
      const value = styles.getPropertyValue(token);
      if (host.style.getPropertyValue(token) !== value) host.style.setProperty(token, value);
    }
    const background = styles.getPropertyValue('--st-background-color').trim();
    if (/^#[0-9a-f]{6}$/i.test(background)) {
      const rgb = [1, 3, 5].map(index => parseInt(background.slice(index, index + 2), 16));
      const luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2];
      const appearance = luminance > 128 ? 'light' : 'dark';
      updateIcon(appearance);
      reportAppearance(appearance);
    }
  }

  function choose(mode, {focus = true} = {}) {
    if (busy) return;
    busy = true;
    status.textContent = '';
    document.documentElement.dataset.floodlabThemeBridge = 'true';
    if (focus) window.__floodlabThemeFocus = true;
    if (!nativeChoice(mode)) nativeMenu()?.click();
    const started = performance.now();

    const apply = () => {
      if (disposed) return;
      const target = nativeChoice(mode);
      if (target) {
        reportAppearance(mode.toLowerCase());
        target.click();
        updateIcon(mode.toLowerCase());
        if (document.querySelector('[data-testid="stMainMenuList"]')) nativeMenu()?.click();
        delete document.documentElement.dataset.floodlabThemeBridge;
        busy = false;
        if (focus) restoreFocus();
      } else if (performance.now() - started < 1500) {
        timer = setTimeout(apply, 25);
      } else {
        delete document.documentElement.dataset.floodlabThemeBridge;
        busy = false;
        status.textContent = 'Choose Theme in the menu.';
      }
    };
    apply();
  }

  button.onclick = () => choose(control.dataset.appearance === 'dark' ? 'Light' : 'Dark');
  sync();
  if (window.__floodlabThemeFocus) restoreFocus();
  const observer = new MutationObserver(() => { if (!busy) sync(); });
  observer.observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['style']
  });
  const systemAppearance = matchMedia('(prefers-color-scheme: dark)');
  const systemChanged = () => requestAnimationFrame(() => { if (!disposed) sync(); });
  systemAppearance.addEventListener('change', systemChanged);

  const cleanup = () => {
    if (root.activeElement === button) window.__floodlabThemeFocus = true;
    disposed = true;
    clearTimeout(timer);
    clearTimeout(focusTimer);
    observer.disconnect();
    systemAppearance.removeEventListener('change', systemChanged);
    host.remove();
    delete document.documentElement.dataset.floodlabThemeBridge;
    if (window.__floodlabThemeCleanup === cleanup) delete window.__floodlabThemeCleanup;
  };
  window.__floodlabThemeCleanup = cleanup;

  // Streamlit may save System before mounting; apply our dark default only once.
  const defaultAppliedKey = 'floodlab-dark-default-v1';
  if (!localStorage.getItem(defaultAppliedKey)) {
    let saved;
    try { saved = JSON.parse(localStorage.getItem(cacheKey)); }
    catch { saved = null; }
    localStorage.setItem(defaultAppliedKey, 'true');
    if (!['Light', 'Dark'].includes(saved)) choose('Dark', {focus: false});
  }
  return cleanup;
}
