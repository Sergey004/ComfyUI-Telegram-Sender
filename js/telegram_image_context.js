import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

// Fail closed unless the one saved output exactly matches the displayed preview.
function currentImage(node) {
  const images = app.nodeOutputs?.[String(node?.id)]?.images;
  if (!Array.isArray(images) || images.length !== 1 || !Array.isArray(node?.imgs) || node.imgs.length !== 1) return null;
  const image = images[0];
  if (!image || image.type !== 'output' || typeof image.filename !== 'string' || !image.filename || typeof image.subfolder !== 'string') return null;
  if (!/\.(png|jpe?g|webp)$/i.test(image.filename) || /[\\/\0]/.test(image.filename) || image.filename === '.' || image.filename === '..' || image.subfolder.startsWith('/') || /[\\\0]/.test(image.subfolder) || (image.subfolder && image.subfolder.split('/').some(part => !part || part === '.' || part === '..'))) return null;
  try {
    const base = new URL(window.location.href), view = new URL(api.apiURL('/view'), base);
    const url = new URL(node.imgs[0]?.src, base);
    if (view.origin !== base.origin || url.origin !== base.origin || url.pathname !== view.pathname || url.username || url.password || url.hash) return null;
    const allowed = new Set(['filename', 'subfolder', 'type', 'rand', 'metadata', 'id']);
    for (const key of url.searchParams.keys()) if (!allowed.has(key) || url.searchParams.getAll(key).length !== 1) return null;
    if (url.searchParams.get('filename') !== image.filename || (url.searchParams.get('subfolder') || '') !== image.subfolder || url.searchParams.get('type') !== 'output') return null;
    return { filename: image.filename, subfolder: image.subfolder, type: 'output' };
  } catch { return null; }
}
function sameImage(a, b) {
  return !!a && !!b && a.filename === b.filename && a.subfolder === b.subfolder && a.type === b.type;
}
// getRandomValues is available on insecure HTTP origins, unlike randomUUID.
function requestId() {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
async function sendOnce(payload) {
  try {
    const response = await api.fetchApi('/telegram_sender/image_context', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => null);
    if (typeof data?.status === 'string') return data.status;
  } catch { /* Delivery may already have happened. */ }
  return 'unknown';
}
function openDialog(node, image) {
  const dialog = document.createElement('dialog');
  dialog.setAttribute('aria-label', '发送图片到 Telegram');
  const title = document.createElement('h2'); title.textContent = '发送图片到 Telegram';
  const info = document.createElement('p'); info.textContent = '将使用 Telegram Sender 当前已保存的默认目标发送原图。请先保存设置。';
  const filename = document.createElement('p'); filename.textContent = `图片：${image.subfolder ? image.subfolder + '/' : ''}${image.filename}`;
  const preview = document.createElement('img'); preview.alt = '待发送原图'; preview.style.maxWidth = '100%'; preview.style.maxHeight = '50vh';
  const status = document.createElement('p'); status.setAttribute('role', 'status'); status.textContent = '正在加载原图…';
  const send = document.createElement('button'); send.textContent = '确认发送'; send.disabled = true;
  const cancel = document.createElement('button'); cancel.textContent = '取消';
  dialog.append(title, info, filename, preview, status, send, cancel); document.body.append(dialog);
  let submitted = false, inFlight = false, digest = null, objectUrl = null, closed = false;
  cancel.addEventListener('click', () => { if (!inFlight) dialog.close(); });
  dialog.addEventListener('cancel', event => { if (inFlight) event.preventDefault(); });
  dialog.addEventListener('close', () => { closed = true; if (objectUrl) URL.revokeObjectURL(objectUrl); dialog.remove(); }, { once: true });
  (async () => {
    try {
      const response = await api.fetchApi('/telegram_sender/image_context/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(image),
      });
      if (!response.ok) throw new Error('preview');
      const sha = response.headers.get('X-Image-SHA256');
      if (!/^[0-9a-f]{64}$/.test(sha || '')) throw new Error('digest');
      const blob = await response.blob();
      if (closed) return;
      objectUrl = URL.createObjectURL(blob); preview.src = objectUrl;
      await preview.decode();
      if (closed) return;
      digest = sha; send.disabled = false; status.textContent = '确认发送此图片？';
    } catch { if (!closed) status.textContent = '无法加载原图，请关闭并重新选择。'; }
  })();
  send.addEventListener('click', async () => {
    if (submitted) return;
    if (!sameImage(image, currentImage(node))) {
      status.textContent = '图片已变化或不再可准确识别，请关闭并重新选择。'; send.disabled = true; return;
    }
    let id;
    try { id = requestId(); }
    catch { status.textContent = '安全请求 ID 不可用，无法发送。'; send.disabled = true; return; }
    submitted = true; inFlight = true; send.disabled = true; cancel.disabled = true;
    status.textContent = '发送中…';
    const result = await sendOnce({ ...image, sha256: digest, idempotency_key: id });
    const messages = {
      sent: '发送成功。', failed: '发送失败，请检查服务端状态。',
      sending: '请求仍在发送中，可能已经送达；请先核实，切勿重复发送。',
      unknown: '发送结果未知，可能已经送达；请先核实，切勿直接重复发送。',
      forbidden: '请求被拒绝，请检查身份和来源授权。', invalid_request: '请求无效，请检查图片。',
      key_conflict: '请求 ID 冲突，请先核实是否已发送。',
      rate_limited: '请求过于频繁，未发送。', capacity_exhausted: '服务容量已满，未发送。',
      invalid_image: '图片不存在、已过期或不受支持。', busy: '发送繁忙，请稍后重试。',
      image_changed: '图片内容已变化，未发送。请关闭并重新确认原图。',
      multi_user_unsupported: '多用户模式暂不支持此功能。', unauthorized: '未授权。', settings_unavailable: '无法读取设置，请检查服务端。',
      missing_settings: '未配置默认目标或 Bot Token，请先在原 Telegram Sender 设置中填写并保存。',
    };
    status.textContent = messages[result] || messages.unknown;
    inFlight = false; cancel.disabled = false; cancel.textContent = '关闭';
  });
  dialog.showModal();
}
app.registerExtension({
  name: 'TelegramSender.ImageContextMenu',
  getNodeMenuItems(node) {
    const image = currentImage(node);
    if (!image) return [];
    return [{ content: '发送图片到 Telegram…', callback: () => {
      if (sameImage(image, currentImage(node))) openDialog(node, image);
    } }];
  },
});
