(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const views = ['access', 'upload', 'uploading', 'processing', 'failed', 'done'];
  const player = $('video-player');
  const lightbox = $('lightbox');
  let selectedFile = null;
  let generation = 0;
  let uploadRequest = null;
  let pollController = null;
  let pollTimer = null;
  let elapsedTimer = null;
  let pendingSeek = null;
  let seekGeneration = 0;
  let activeView = 'upload';

  function showView(name, heading) {
    activeView = name;
    views.forEach((view) => { $(`${view}-view`).hidden = view !== name; });
    if (heading) $(heading).focus({ preventScroll: true });
  }

  function needAccess() {
    stopWork();
    $('access-error').hidden = true;
    showView('access', 'access-title');
    $('access-code').focus();
  }

  function stopWork() {
    generation += 1;
    clearTimeout(pollTimer);
    clearInterval(elapsedTimer);
    if (uploadRequest) uploadRequest.abort();
    if (pollController) pollController.abort();
    uploadRequest = null;
    pollController = null;
    pendingSeek = null;
    seekGeneration += 1;
  }

  function clearResult() {
    if (lightbox.open) lightbox.close();
    $('lightbox-image').removeAttribute('src');
    player.pause();
    player.removeAttribute('src');
    player.load();
    $('violations-list').replaceChildren();
    $('api-json').textContent = '';
    $('api-json').parentElement.open = false;
    $('player-note').hidden = true;
  }

  function reset(changeHash = true) {
    stopWork();
    clearResult();
    selectedFile = null;
    $('file-input').value = '';
    $('file-info').textContent = 'Файл не выбран';
    $('selection-error').hidden = true;
    $('analyze-button').disabled = true;
    if (changeHash) history.replaceState(null, '', location.pathname + location.search);
    showView('upload');
    $('choose-file').focus({ preventScroll: true });
  }

  function fail(message) {
    stopWork();
    $('error-message').textContent = message;
    showView('failed', 'failed-title');
  }

  function errorText(payload, fallback) {
    const detail = payload && (payload.detail || payload.error);
    if (typeof detail === 'string' && detail.trim()) return detail;
    return detail ? JSON.stringify(detail) : fallback;
  }

  function formatTime(seconds) {
    const tenths = Math.round(Math.max(0, Number(seconds) || 0) * 10);
    return `${String(Math.floor(tenths / 600)).padStart(2, '0')}:${String(Math.floor(tenths / 10) % 60).padStart(2, '0')}.${tenths % 10}`;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} Б`;
    const unit = bytes < 1024 ** 2 ? 'КБ' : bytes < 1024 ** 3 ? 'МБ' : 'ГБ';
    const divisor = unit === 'КБ' ? 1024 : unit === 'МБ' ? 1024 ** 2 : 1024 ** 3;
    return `${(bytes / divisor).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} ${unit}`;
  }

  function selectFiles(files) {
    if (activeView !== 'upload') return;
    $('selection-error').hidden = true;
    if (files.length !== 1) {
      $('selection-error').textContent = 'Выберите один видеофайл.';
      $('selection-error').hidden = false;
      return;
    }
    selectedFile = files[0];
    $('file-info').textContent = `${selectedFile.name} · ${formatSize(selectedFile.size)}`;
    $('analyze-button').disabled = selectedFile.size === 0;
    if (!selectedFile.size) {
      $('selection-error').textContent = 'Файл пуст. Выберите другое видео.';
      $('selection-error').hidden = false;
    }
  }

  $('choose-file').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (event) => {
    if (event.target.files.length) selectFiles(event.target.files);
  });
  let dragDepth = 0;
  const dropZone = $('drop-zone');
  dropZone.addEventListener('dragenter', (event) => {
    event.preventDefault();
    dragDepth += 1;
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });
  dropZone.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove('drag-over');
    $('file-input').value = '';
    selectFiles(event.dataTransfer.files);
  });
  // Prevent a dropped file outside the upload area from replacing the page.
  for (const type of ['dragover', 'drop']) {
    window.addEventListener(type, (event) => {
      if (Array.from(event.dataTransfer?.types || []).includes('Files')) event.preventDefault();
    });
  }

  $('upload-form').addEventListener('submit', (event) => {
    event.preventDefault();
    if (!selectedFile || !selectedFile.size || activeView !== 'upload') return;
    stopWork();
    const token = generation;
    const xhr = new XMLHttpRequest();
    uploadRequest = xhr;
    $('upload-name').textContent = `${selectedFile.name} · ${formatSize(selectedFile.size)}`;
    $('upload-progress').value = 0;
    $('upload-progress-text').textContent = 'Загружено 0 %';
    showView('uploading', 'uploading-title');
    xhr.open('POST', '/ui/analyze');
    xhr.responseType = 'json';
    xhr.upload.addEventListener('progress', (progress) => {
      if (token !== generation) return;
      if (progress.lengthComputable) {
        const percent = Math.min(100, Math.round(progress.loaded / progress.total * 100));
        $('upload-progress').value = percent;
        $('upload-progress-text').textContent = percent === 100
          ? 'Файл передан. Сервер проверяет видео…' : `Загружено ${percent} %`;
      } else {
        $('upload-progress').removeAttribute('value');
        $('upload-progress-text').textContent = `Передано ${formatSize(progress.loaded)}`;
      }
    });
    xhr.addEventListener('load', () => {
      if (token !== generation) return;
      uploadRequest = null;
      const payload = xhr.response;
      if (xhr.status === 401) {
        needAccess();
      } else if (xhr.status < 200 || xhr.status >= 300) {
        fail(errorText(payload, `Не удалось загрузить видео. Код ответа: ${xhr.status}.`));
      } else if (!payload || typeof payload.job_id !== 'string' || !payload.job_id) {
        fail('Сервер не вернул номер задания. Попробуйте загрузить видео ещё раз.');
      } else {
        history.replaceState(null, '', `#job=${encodeURIComponent(payload.job_id)}`);
        startJob(payload.job_id);
      }
    });
    xhr.addEventListener('error', () => {
      if (token === generation) fail('Не удалось передать видео: нет связи с сервером. Проверьте подключение и повторите загрузку.');
    });
    const form = new FormData();
    form.append('file', selectedFile);
    xhr.send(form);
  });

  function startJob(jobId) {
    stopWork();
    clearResult();
    const token = generation;
    const startedAt = Date.now();
    $('elapsed').textContent = '0';
    $('connection-note').hidden = true;
    showView('processing', 'processing-title');
    elapsedTimer = setInterval(() => {
      $('elapsed').textContent = String(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    poll(jobId, token);
  }

  async function poll(jobId, token) {
    if (token !== generation) return;
    const controller = new AbortController();
    pollController = controller;
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(`/ui/jobs/${encodeURIComponent(jobId)}`, {
        signal: controller.signal, cache: 'no-store', headers: { Accept: 'application/json' }
      });
      if (token !== generation) return;
      if (response.status === 401) {
        needAccess();
        return;
      }
      if (!response.ok) {
        if (response.status >= 500 || [408, 429].includes(response.status)) throw new Error('retry');
        const payload = await response.json().catch(() => null);
        if (token === generation) fail(errorText(payload, `Не удалось получить задание. Код ответа: ${response.status}.`));
        return;
      }
      const payload = await response.json();
      if (token !== generation) return;
      $('connection-note').hidden = true;
      if (payload.status === 'done') {
        if (!['violation', 'no_violation'].includes(payload.verdict) || !Array.isArray(payload.violations)) {
          fail('Сервер вернул результат в неизвестном формате.');
          return;
        }
        clearInterval(elapsedTimer);
        renderResult(payload, jobId);
        return;
      }
      if (payload.status === 'failed') {
        fail(errorText(payload, 'Анализ завершился ошибкой. Попробуйте другое видео.'));
        return;
      }
      if (payload.status !== 'processing') throw new Error('retry');
    } catch {
      if (token !== generation) return;
      $('connection-note').hidden = false;
    } finally {
      clearTimeout(timeout);
      if (pollController === controller) pollController = null;
    }
    if (token === generation) pollTimer = setTimeout(() => poll(jobId, token), 2000);
  }

  function seekTo(seconds) {
    pendingSeek = Math.max(0, Number(seconds) || 0);
    seekGeneration += 1;
    if (player.readyState >= 1) applySeek();
    else {
      $('player-note').textContent = 'Загружаем видео для перехода к выбранному моменту…';
      $('player-note').hidden = false;
      player.load();
    }
  }

  function applySeek() {
    if (pendingSeek === null) return;
    const target = pendingSeek;
    const token = seekGeneration;
    pendingSeek = null;
    try {
      player.currentTime = Number.isFinite(player.duration) ? Math.min(target, player.duration) : target;
      $('player-note').hidden = true;
      player.play().catch(() => {
        if (token !== seekGeneration) return;
        $('player-note').textContent = 'Нужный момент выбран. Нажмите кнопку воспроизведения на видео.';
        $('player-note').hidden = false;
      });
      player.scrollIntoView({ block: 'nearest' });
    } catch {
      $('player-note').textContent = 'Не удалось перейти к этому моменту. Попробуйте после загрузки видео.';
      $('player-note').hidden = false;
    }
  }
  player.addEventListener('loadedmetadata', applySeek);
  player.addEventListener('error', () => {
    if (!player.getAttribute('src')) return;
    $('player-note').textContent = 'Не удалось воспроизвести видео. Проверьте подключение и поддержку формата в браузере.';
    $('player-note').hidden = false;
  });

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderResult(payload, jobId) {
    const clean = payload.verdict === 'no_violation';
    $('verdict-banner').className = `verdict ${clean ? 'success' : 'warning'}`;
    $('verdict-title').textContent = clean ? 'Нарушение не обнаружено' : 'Обнаружены признаки возможных нарушений';
    $('no-violation-note').hidden = !clean;
    player.src = `/ui/jobs/${encodeURIComponent(jobId)}/video`;
    $('violations-list').replaceChildren();
    $('violations-section').hidden = !payload.violations.length;
    $('violation-count').textContent = String(payload.violations.length);
    payload.violations.forEach((violation, index) => {
      const card = element('article', 'panel violation-card');
      const heading = element('div', 'card-heading');
      const title = element('h3', '', violation.type);
      title.id = `violation-${index}`;
      card.setAttribute('aria-labelledby', title.id);
      const time = element('a', 'timecode', formatTime(violation.timecode_sec));
      time.href = `${player.getAttribute('src')}#t=${Math.max(0, Number(violation.timecode_sec) || 0)}`;
      time.setAttribute('aria-label', `Воспроизвести с ${formatTime(violation.timecode_sec)}`);
      time.addEventListener('click', (event) => { event.preventDefault(); seekTo(violation.timecode_sec); });
      heading.append(title, time);
      card.append(heading, element('p', 'description', violation.description));
      const grid = element('div', 'evidence-grid');
      (violation.evidence_frames || []).slice(0, 3).forEach((frame) => {
        const timestamp = formatTime(frame.timestamp_sec);
        const figure = element('figure');
        const seek = element('button', 'frame-button');
        seek.type = 'button';
        seek.setAttribute('aria-label', `Перейти к кадру ${timestamp}`);
        const img = element('img');
        img.src = `data:image/jpeg;base64,${frame.image_base64}`;
        img.alt = `${violation.type}: кадр ${timestamp}`;
        img.loading = 'lazy';
        seek.append(img);
        seek.addEventListener('click', () => seekTo(frame.timestamp_sec));
        const caption = element('figcaption');
        const zoom = element('button', 'zoom-button', 'увеличить');
        zoom.type = 'button';
        zoom.setAttribute('aria-label', `Увеличить кадр ${timestamp}`);
        zoom.addEventListener('click', () => {
          $('lightbox-image').src = img.src;
          $('lightbox-image').alt = img.alt;
          $('lightbox-title').textContent = `Кадр ${timestamp}`;
          lightbox.showModal();
          $('close-lightbox').focus();
        });
        caption.append(element('span', '', timestamp), zoom);
        figure.append(seek, caption);
        grid.append(figure);
      });
      card.append(grid);
      $('violations-list').append(card);
    });
    $('api-json').textContent = JSON.stringify(payload, (key, value) =>
      key === 'image_base64' && typeof value === 'string' ? `${value.slice(0, 40)}…` : value, 2);
    showView('done', 'verdict-title');
  }

  $('close-lightbox').addEventListener('click', () => lightbox.close());
  lightbox.addEventListener('click', (event) => {
    if (event.target === lightbox || event.target === $('lightbox-image')) lightbox.close();
  });
  lightbox.addEventListener('close', () => $('lightbox-image').removeAttribute('src'));
  document.querySelectorAll('[data-reset]').forEach((button) => button.addEventListener('click', () => reset()));

  function restoreHash() {
    const jobId = new URLSearchParams(location.hash.slice(1)).get('job');
    if (jobId) startJob(jobId);
    else reset(false);
  }
  window.addEventListener('hashchange', restoreHash);

  $('access-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = event.submitter;
    if (button) button.disabled = true;
    try {
      const response = await fetch('/ui/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: $('access-code').value.trim() })
      });
      if (!response.ok) {
        $('access-error').textContent = response.status === 401 ? 'Неверный код доступа.' : `Ошибка входа. Код ответа: ${response.status}.`;
        $('access-error').hidden = false;
        return;
      }
      $('access-code').value = '';
      restoreHash();
    } catch {
      $('access-error').textContent = 'Нет связи с сервером. Повторите попытку.';
      $('access-error').hidden = false;
    } finally {
      if (button) button.disabled = false;
    }
  });

  (async () => {
    try {
      const session = await (await fetch('/ui/session', { cache: 'no-store' })).json();
      if (session.required && !session.ok) { needAccess(); return; }
    } catch { /* server unreachable: the upload flow reports it */ }
    if (location.hash) restoreHash();
  })();
})();
