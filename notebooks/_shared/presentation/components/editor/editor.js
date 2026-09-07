// Primitive Rateless editor: input gestures and Python-owned state rendering.
function render({ model, el }) {
  // VS Code focuses the first focusable descendant of an output when its
  // current activeElement is outside that output. Make that target neutral.
  el.tabIndex = 0;
  function onPointerDown(event) {
    el.classList.add('pr2-pointer-focus');
    // Focus before the browser starts text selection; never blur/restore ranges.
    if (!event.target.closest('input')) el.focus({preventScroll: true});
  }
  function onKeyboard(event) {
    if (event.key === 'Tab') el.classList.remove('pr2-pointer-focus');
  }
  function onClick(event) {
    // Keep output focus after a button action without leaving the button focused.
    if (event.detail > 0 && event.target.closest('button')) el.focus({preventScroll: true});
    const label = event.target.closest('label');
    const selection = el.ownerDocument.getSelection();
    if (label && !event.target.closest('input, button') && selection &&
        !selection.isCollapsed && selection.containsNode(label, true)) {
      event.preventDefault();
    }
  }
  el.addEventListener('pointerdown', onPointerDown, true);
  el.addEventListener('click', onClick);
  el.ownerDocument.addEventListener('keydown', onKeyboard, true);
  el.classList.add("pr2-host");
  el.innerHTML = model.get("template");

  const q = selector => el.querySelector(selector);
  el.querySelectorAll('.pr2-grid').forEach(grid => {
    if (!grid.closest('.pr2-codeword-scroll')) grid.tabIndex = 0;
  });
  el.querySelectorAll('label').forEach(label => {
    if (!label.querySelector('input[type="number"]')) return;
    [...label.childNodes].forEach(node => {
      if (node.nodeType !== Node.TEXT_NODE || !node.textContent.trim()) return;
      const text = document.createElement('span');
      text.className = 'pr2-number-label';
      text.textContent = node.textContent;
      node.replaceWith(text);
    });
  });
  // Explicit steppers stay visible in every notebook/browser theme.
  el.querySelectorAll('input[type="number"]').forEach(input => {
    const wrapper = document.createElement('span');
    wrapper.className = 'pr2-number';
    input.before(wrapper);
    wrapper.append(input);
    const arrows = document.createElement('span');
    arrows.className = 'pr2-number-arrows';
    for (const [direction, glyph, name] of [[1, '▴', 'Increase'], [-1, '▾', 'Decrease']]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = glyph;
      const label = wrapper.closest('label').textContent.trim();
      button.setAttribute('aria-label', `${name} ${label}`);
      button.title = `${name} ${label}`;
      button.onclick = () => {
        if (input.disabled) return;
        if (direction > 0) input.stepUp(); else input.stepDown();
        input.dispatchEvent(new Event('input', {bubbles: true}));
        input.dispatchEvent(new Event('change', {bubbles: true}));
      };
      arrows.append(button);
    }
    wrapper.append(arrows);
  });
  let mode = 'flip', down = false, current = null, timer = null;
  const send = event => model.send(event);
  function stop() {
    down = false; current = null;
    if (timer !== null) clearInterval(timer);
    timer = null;
  }
  function paint(index) {
    if (index === null) return;
    const amount = mode === 'suppress' ? (q('.pr2-erase').getAttribute('aria-pressed') === 'true' ? 'inf' : q('.pr2-suppress-db').value)
      : q(mode === 'flip' ? '.pr2-flip-rate' : '.pr2-restore-rate').value;
    send({action:'paint', index, mode, amount});
  }
  function cells(selector, values, kind, colors) {
    const host = q(selector);
    if (host.children.length !== values.length) {
      stop();
      host.replaceChildren(...values.map((_, index) => {
        const cell = document.createElement('button');
        cell.type = 'button'; cell.className = `pr2-cell pr2-${kind}-cell`;
        cell.dataset.index = index; cell.disabled = kind === 'y' || kind === 'channel';
        return cell;
      }));
    }
    [...host.children].forEach((cell, index) => {
      const value = values[index];
      const text = kind === 'x' || kind === 'y' ? String(value) : kind === 'channel' ? (value > 0 ? '+1' : '\u22121') :
        `${value >= 0 ? '+' : ''}${Math.abs(value) >= 1000 ? value.toExponential(1) : value.toFixed(2)}`;
      if (cell.textContent !== text) cell.textContent = text;
      if (colors) {cell.style.background = colors[index]; cell.style.color = '#171717';}
      cell.title = `i=${index}, ${kind === 'x' || kind === 'y' ? 'bit' : 'LLR'}=${value}`;
      cell.setAttribute('aria-label', `${kind} bit ${index}: ${value}`);
    });
  }
  function drawMode() {
    el.querySelectorAll('[data-mode]').forEach(button => {
      button.classList.toggle('active', button.dataset.mode === mode);
      button.setAttribute('aria-checked', String(button.dataset.mode === mode));
    });
    for (const name of ['flip', 'suppress', 'restore']) q(`.pr2-${name}-param`).hidden = mode !== name;
    q('.pr2-erase-param').hidden = mode !== 'suppress';
  }
  function update() {
    const s = model.get('state');
    const polynomial = q('.pr2-polynomial');
    if (polynomial.dataset.value !== s.polynomial) {
      polynomial.dataset.value = s.polynomial;
      polynomial.replaceChildren(...s.polynomial.split(/\^(\d+)/).map((part, index) => {
      if (index % 2 === 0) return document.createTextNode(part);
      const exponent = document.createElement('sup');
      exponent.textContent = part;
      return exponent;
      }));
    }
    const setText = (selector, value) => {
      const node = q(selector);
      if (node.textContent !== String(value)) node.textContent = value;
    };
    setText('.pr2-k', s.x.length);
    for (const [selector, value] of [['.pr2-n', s.n], ['.pr2-t', s.iterations]]) {
      const input = q(selector);
      if (el.ownerDocument.activeElement !== input && input.value !== String(value)) {
        input.value = value;
      }
    }
    setText('.pr2-status', s.error || '');
    q('.pr2-status').hidden = !s.error;
    cells('.pr2-x', s.x, 'x'); cells('.pr2-y', s.y, 'y');
    cells('.pr2-channel', s.channel_llr, 'channel');
    cells('.pr2-yprime', s.edited, 'yprime', s.colors);
  }
  q('.pr2-x').onclick = event => {
    const target = event.target.closest('.pr2-x-cell');
    if (target) {stop(); send({action:'flip_x', index:Number(target.dataset.index)});}
  };
  q('.pr2-random').onclick = () => {stop(); send({action:'random'});};
  q('.pr2-reset').onclick = () => {stop(); send({action:'reset'});};
  q('.pr2-global').onclick = () => send({action:'noise', sigma:q('.pr2-global-sigma').value});
  q('.pr2-segmented').onclick = event => {
    const target = event.target.closest('[data-mode]');
    if (target) {stop(); mode = target.dataset.mode; drawMode();}
  };
  q('.pr2-erase').onclick = () => {
    const button = q('.pr2-erase');
    const enabled = button.getAttribute('aria-pressed') !== 'true';
    button.setAttribute('aria-pressed', String(enabled));
    const input = q('.pr2-suppress-db');
    input.disabled = enabled;
    input.parentElement.querySelectorAll('button').forEach(button => {
      button.disabled = input.disabled;
    });
  };
  const configure = () => {stop(); send({action:'configure', n:q('.pr2-n').value, iterations:q('.pr2-t').value});};
  q('.pr2-n').onchange = configure; q('.pr2-t').onchange = configure;
  q('.pr2-yprime').onpointerdown = event => {
    const target = event.target.closest('.pr2-yprime-cell');
    if (!target || event.button !== 0) return;
    event.preventDefault(); stop(); down = true;
    current = Number(target.dataset.index); paint(current);
    timer = setInterval(() => paint(current), 100);
  };
  q('.pr2-yprime').onkeydown = event => {
    const target = event.target.closest('.pr2-yprime-cell');
    if (target && [' ', 'Enter'].includes(event.key)) {event.preventDefault(); paint(Number(target.dataset.index));}
  };
  function pointerMove(event) {
    if (!down) return;
    if (event.buttons !== 1) {stop(); return;}
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('.pr2-yprime-cell');
    const next = target && el.contains(target) ? Number(target.dataset.index) : null;
    if (next !== current) {current = next; paint(current);}
  }
  document.addEventListener('pointermove', pointerMove);
  document.addEventListener('pointerup', stop);
  document.addEventListener('pointercancel', stop);
  document.addEventListener('visibilitychange', stop);
  window.addEventListener('blur', stop);
  model.on('change:state', update);
  update(); drawMode();
  const hint = q('.pr2-drag-hint');
  const hintText = q('.pr2-hint-text');
  function resizeHint() {
    const overflow = Math.max(0, hintText.scrollWidth - hint.clientWidth);
    hint.style.setProperty('--pr2-hint-shift', `${-overflow}px`);
    hint.style.setProperty('--pr2-hint-duration', `${Math.max(5, overflow / 25 + 3)}s`);
    hint.classList.toggle('pr2-scrolling', overflow > 1);
  }
  const hintObserver = new ResizeObserver(resizeHint);
  hintObserver.observe(hint);
  hintObserver.observe(hintText);
  resizeHint();
  return () => {
    el.removeEventListener('click', onClick);
    el.removeEventListener('pointerdown', onPointerDown, true);
    el.ownerDocument.removeEventListener('keydown', onKeyboard, true);
    hintObserver.disconnect();
    stop(); model.off('change:state', update);
    document.removeEventListener('pointermove', pointerMove);
    document.removeEventListener('pointerup', stop);
    document.removeEventListener('pointercancel', stop);
    document.removeEventListener('visibilitychange', stop);
    window.removeEventListener('blur', stop);
  };
}
export default { render };
