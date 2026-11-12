const modelList = document.querySelector('#model-list');
const textarea = document.querySelector('#text-input');
const countLabel = document.querySelector('#char-count');
const predictButton = document.querySelector('#predict-button');
const resultsSection = document.querySelector('#results-section');
const resultsGrid = document.querySelector('#results-grid');
const notice = document.querySelector('#notice');
const historySection = document.querySelector('#history-section');
let lastPrediction = null;

let models = [];

textarea.addEventListener('input', () => {
  countLabel.textContent = `${textarea.value.length} / 200`;
});

function showNotice(message) {
  notice.textContent = message;
  notice.hidden = false;
  window.clearTimeout(showNotice.timer);
  showNotice.timer = window.setTimeout(() => { notice.hidden = true; }, 4500);
}

function checkbox(model, checked) {
  const label = document.createElement('label');
  label.className = `model-option${model.available ? '' : ' disabled'}`;

  const input = document.createElement('input');
  input.type = 'checkbox';
  input.name = 'models';
  input.value = model.key;
  input.checked = checked && model.available;
  input.disabled = !model.available;

  const title = document.createElement('span');
  title.className = 'model-option-label';
  title.textContent = model.name;

  const state = document.createElement('span');
  state.className = 'model-state';
  state.textContent = model.available ? '可用' : '需权重 / 依赖';

  label.append(input, title, state);
  return label;
}

async function loadModels() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error('无法读取模型状态');
    const payload = await response.json();
    models = payload.models;
    modelList.replaceChildren();
    for (const model of models) {
      const defaultChecked = ['bert', 'fasttext'].includes(model.key);
      modelList.append(checkbox(model, defaultChecked));
    }
  } catch (error) {
    modelList.textContent = '模型服务连接失败，请确认后端已启动。';
    showNotice(error.message);
  }
}

function probabilityRow(item) {
  const row = document.createElement('div');
  row.className = 'probability-row';
  const label = document.createElement('span');
  label.className = 'probability-label';
  label.textContent = item.label;
  label.title = item.label;
  const track = document.createElement('span');
  track.className = 'bar-track';
  const fill = document.createElement('span');
  fill.className = 'bar-fill';
  fill.style.width = `${Math.min(100, item.probability * 100)}%`;
  track.append(fill);
  const value = document.createElement('span');
  value.className = 'probability-value';
  value.textContent = `${(item.probability * 100).toFixed(1)}%`;
  row.append(label, track, value);
  return row;
}

function resultCard(result) {
  const card = document.createElement('article');
  card.className = 'result-card';

  const head = document.createElement('header');
  head.className = 'result-card-head';
  const name = document.createElement('h3');
  name.textContent = result.name;
  const prediction = document.createElement('div');
  prediction.className = 'prediction';
  const predicted = document.createElement('strong');
  predicted.textContent = result.predicted_class;
  const confidence = document.createElement('span');
  confidence.textContent = `置信度 ${(result.confidence * 100).toFixed(1)}%`;
  prediction.append(predicted, confidence);
  const meta = document.createElement('div');
  meta.className = 'result-meta';
  const latency = document.createElement('span');
  latency.textContent = `本次耗时 ${result.latency_ms.toFixed(1)} ms`;
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'copy-button';
  copy.textContent = '复制结果';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      copy.textContent = '已复制';
      window.setTimeout(() => { copy.textContent = '复制结果'; }, 1200);
    } catch (_) {
      showNotice('浏览器未允许访问剪贴板。');
    }
  });
  meta.append(latency, copy);
  head.append(name, prediction, meta);

  const list = document.createElement('div');
  list.className = 'probability-list';
  const initialCount = 8;
  result.probabilities.slice(0, initialCount).forEach(item => list.append(probabilityRow(item)));
  card.append(head, list);

  if (result.probabilities.length > initialCount) {
    const more = document.createElement('button');
    more.type = 'button';
    more.className = 'more-button';
    more.textContent = '展开全部 26 个类别';
    more.addEventListener('click', () => {
      const expanded = more.dataset.expanded === 'true';
      if (expanded) {
        list.replaceChildren(...result.probabilities.slice(0, initialCount).map(probabilityRow));
        more.textContent = '展开全部 26 个类别';
      } else {
        list.replaceChildren(...result.probabilities.map(probabilityRow));
        more.textContent = '收起类别';
      }
      more.dataset.expanded = String(!expanded);
    });
    card.append(more);
  }
  return card;
}

function renderHistory() {
  const history = JSON.parse(localStorage.getItem('stc-history') || '[]');
  const historyList = document.querySelector('#history-list');
  historySection.hidden = history.length === 0;
  historyList.replaceChildren();
  history.forEach(item => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'history-item';
    const text = document.createElement('strong');
    text.textContent = item.text;
    const summary = document.createElement('span');
    summary.textContent = `${item.models.length} 个模型 · ${item.prediction}`;
    button.append(text, summary);
    button.addEventListener('click', () => {
      textarea.value = item.text;
      textarea.dispatchEvent(new Event('input'));
      const keys = new Set(item.models);
      document.querySelectorAll('input[name="models"]').forEach(input => { input.checked = keys.has(input.value); });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    historyList.append(button);
  });
}

function saveHistory(payload) {
  const oldHistory = JSON.parse(localStorage.getItem('stc-history') || '[]');
  const summary = payload.results.map(item => item.predicted_class).join(' / ');
  const history = [{
    text: payload.text,
    models: payload.results.map(item => item.key),
    prediction: summary,
  }, ...oldHistory.filter(item => item.text !== payload.text)].slice(0, 6);
  localStorage.setItem('stc-history', JSON.stringify(history));
  renderHistory();
}

function renderAgreement(results) {
  const counts = new Map();
  results.forEach(result => counts.set(result.predicted_class, (counts.get(result.predicted_class) || 0) + 1));
  const [label, votes] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const summary = document.querySelector('#agreement-summary');
  const icon = document.createElement('span');
  icon.className = 'agreement-icon';
  icon.textContent = votes === results.length && results.length > 1 ? '✓' : '↗';
  const message = document.createElement('span');
  message.textContent = results.length === 1
    ? `当前模型最可能的类别：${label}`
    : `${votes} / ${results.length} 个模型把「${label}」排在第一位。各模型概率见下方。`;
  summary.replaceChildren(icon, message);
}

document.querySelector('#select-available').addEventListener('click', () => {
  document.querySelectorAll('input[name="models"]:not(:disabled)').forEach(input => { input.checked = true; });
});

document.querySelector('#clear-input').addEventListener('click', () => {
  textarea.value = '';
  textarea.dispatchEvent(new Event('input'));
  textarea.focus();
});

document.querySelectorAll('.example-chip').forEach(button => {
  button.addEventListener('click', () => {
    textarea.value = button.textContent;
    textarea.dispatchEvent(new Event('input'));
    textarea.focus();
  });
});

document.querySelector('#clear-history').addEventListener('click', () => {
  localStorage.removeItem('stc-history');
  renderHistory();
});

document.querySelector('#export-results').addEventListener('click', () => {
  if (!lastPrediction) return showNotice('请先完成一次预测。');
  const rows = [['text', 'model', 'class_id', 'label', 'probability']];
  lastPrediction.results.forEach(result => result.probabilities.forEach(item => {
    rows.push([lastPrediction.text, result.name, item.class_id, item.label, item.probability]);
  }));
  const csv = rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob(['\ufeff', csv], { type: 'text/csv;charset=utf-8' }));
  link.download = 'short-text-classification-probabilities.csv';
  link.click();
  URL.revokeObjectURL(link.href);
});

predictButton.addEventListener('click', async () => {
  const text = textarea.value.trim();
  const selected = [...document.querySelectorAll('input[name="models"]:checked')].map(input => input.value);
  if (!text) return showNotice('先输入一条短文本。');
  if (!selected.length) return showNotice('至少选择一个可用模型。');

  predictButton.disabled = true;
  predictButton.querySelector('span:first-child').textContent = '模型计算中…';
  notice.hidden = true;
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        models: selected,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '预测失败');
    resultsGrid.replaceChildren(...payload.results.map(resultCard));
    lastPrediction = payload;
    renderAgreement(payload.results);
    document.querySelector('#result-text').textContent = `“${payload.text}”`;
    resultsSection.hidden = false;
    saveHistory(payload);
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    showNotice(error.message);
  } finally {
    predictButton.disabled = false;
    predictButton.querySelector('span:first-child').textContent = '开始预测';
  }
});

loadModels();
renderHistory();

