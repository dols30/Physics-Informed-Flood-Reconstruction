// Native menu selectors match the Streamlit version pinned in app/requirements.txt.
// Forward to native actions so widgets, plots, system mode, and persistence agree.
export default function({ parentElement, setStateValue, data }) {
  window.__floodlabThemeCleanup?.();
  const source = parentElement.querySelector('.theme-switcher');
  const header = document.querySelector('[data-testid="stHeader"]');
  if (!header) return;
  const host = document.createElement('div');
  host.dataset.floodlabThemeControl = 'true';
  const root = host.attachShadow({mode: 'open'});
  for (const style of parentElement.querySelectorAll('style')) root.append(style.cloneNode(true));
  const control = source.cloneNode(true);
  control.style.removeProperty('display');
  root.append(control);
  source.style.display = 'none';
  header.append(host);
  const buttons = [...control.querySelectorAll('[data-theme]')];
  const status = control.querySelector('.theme-status');
  const nativeMenu = () => document.querySelector('[data-testid="stMainMenuButton"]');
  const nativeChoice = mode => document.querySelector(`[data-testid="stMainMenuItem-theme-${mode}"]`);
  const cacheKey = window.__floodlabThemeCacheKey ||= `stActiveTheme-${location.pathname}-v2`;
  let busy = false;
  let disposed = false;
  let timer;
  let focusTimer;
  let reportedAppearance = data?.appearance;

  function reportAppearance(mode) {
    const appearance = mode === 'System'
      ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : mode.toLowerCase();
    if (['light', 'dark'].includes(appearance) && appearance !== reportedAppearance) {
      reportedAppearance = appearance;
      setStateValue('appearance', appearance);
    }
  }

  function restoreFocus(mode) {
    clearTimeout(focusTimer);
    focusTimer = setTimeout(() => {
      if (!disposed && window.__floodlabThemeFocus === mode) {
        buttons.find(button => button.dataset.theme === mode)?.focus({preventScroll: true});
        delete window.__floodlabThemeFocus;
      }
    }, 100);
  }

  function select(mode) {
    if (!['System', 'Light', 'Dark'].includes(mode)) return;
    for (const button of buttons) {
      const selected = button.dataset.theme === mode;
      button.setAttribute('aria-checked', String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
  }

  function sync() {
    const styles = getComputedStyle(source);
    for (const token of ['--st-text-color', '--st-primary-color', '--st-background-color']) {
      const value = styles.getPropertyValue(token);
      if (host.style.getPropertyValue(token) !== value) host.style.setProperty(token, value);
    }
    const background = styles.getPropertyValue('--st-background-color').trim();
    if (/^#[0-9a-f]{6}$/i.test(background)) {
      const rgb = [1, 3, 5].map(index => parseInt(background.slice(index, index + 2), 16));
      const luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2];
      reportAppearance(luminance > 128 ? 'light' : 'dark');
    }
    const selected = document.querySelector('[data-testid="stThemeSwitcher"] [aria-checked="true"]');
    if (selected) select(selected.dataset.testid.split('-').at(-1));
    else {
      try { select(JSON.parse(localStorage.getItem(cacheKey)) || 'Dark'); }
      catch { select('Dark'); }
    }
  }

  function choose(button, {focus = true} = {}) {
    if (busy) return;
    busy = true;
    status.textContent = '';
    document.documentElement.dataset.floodlabThemeBridge = 'true';
    const mode = button.dataset.theme;
    if (focus) window.__floodlabThemeFocus = mode;
    if (!nativeChoice(mode)) nativeMenu()?.click();
    const started = performance.now();

    const apply = () => {
      if (disposed) return;
      const target = nativeChoice(mode);
      if (target) {
        reportAppearance(mode);
        target.click();
        select(mode);
        if (document.querySelector('[data-testid="stMainMenuList"]')) nativeMenu()?.click();
        delete document.documentElement.dataset.floodlabThemeBridge;
        busy = false;
        if (focus) restoreFocus(mode);
      } else if (performance.now() - started < 1500) timer = setTimeout(apply, 25);
      else {
        delete document.documentElement.dataset.floodlabThemeBridge;
        busy = false;
        status.textContent = 'Choose Theme in the menu.';
      }
    };
    apply();
  }

  for (const [index, button] of buttons.entries()) {
    button.onclick = () => choose(button);
    button.onkeydown = event => {
      let next;
      if (['ArrowRight', 'ArrowDown'].includes(event.key)) next = (index + 1) % buttons.length;
      if (['ArrowLeft', 'ArrowUp'].includes(event.key)) next = (index + buttons.length - 1) % buttons.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = buttons.length - 1;
      if (next !== undefined) { event.preventDefault(); choose(buttons[next]); }
    };
  }
  sync();
  const focusMode = window.__floodlabThemeFocus;
  if (focusMode) restoreFocus(focusMode);
  const observer = new MutationObserver(() => { if (!busy) sync(); });
  observer.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['aria-checked', 'style']});
  const systemAppearance = matchMedia('(prefers-color-scheme: dark)');
  const systemChanged = () => {
    if (buttons[0].getAttribute('aria-checked') === 'true') {
      requestAnimationFrame(() => { if (!disposed) sync(); });
    }
  };
  systemAppearance.addEventListener('change', systemChanged);
  const cleanup = () => {
    if (root.activeElement?.dataset.theme) window.__floodlabThemeFocus = root.activeElement.dataset.theme;
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
  // Both native palettes are configured, so explicitly select the initial dark
  // appearance instead of Streamlit's automatic system preference. Respect a
  // visitor's saved choice and avoid moving keyboard focus on first load.
  const defaultAppliedKey = 'floodlab-dark-default-v1';
  if (!localStorage.getItem(defaultAppliedKey)) {
    // Native initialization can save "System" before this component mounts.
    // Apply our default once, keeping explicit saved Light/Dark choices.
    let saved;
    try { saved = JSON.parse(localStorage.getItem(cacheKey)); }
    catch { saved = null; }
    localStorage.setItem(defaultAppliedKey, 'true');
    if (!['Light', 'Dark'].includes(saved)) {
      choose(buttons.find(button => button.dataset.theme === 'Dark'), {focus: false});
    }
  }
  return cleanup;
}
