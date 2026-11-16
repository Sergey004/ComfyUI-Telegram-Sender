"""Isolated right-click sender. Import and call register_routes(server) once.

Only single-user ComfyUI is supported: comfy-user is a selector, not authentication.
The deployment must separately enforce its network/authentication boundary. Idempotency
is process-local and is lost on restart; uncertain sends must never be auto-retried.
"""
import asyncio
import hashlib
import hmac
import os
import re
import stat
import uuid
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, FormData, web

MAX_IMAGE = 20 * 1024 * 1024
MAX_KEYS = 4096
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


class InputError(Exception):
    pass


def _same_origin(request):
    origin = request.headers.get('Origin', '')
    try:
        parsed = urlsplit(origin)
        return (parsed.scheme in ('http', 'https') and parsed.scheme == request.scheme
                and parsed.netloc == request.host and not parsed.username
                and not parsed.password and parsed.path == '' and not parsed.query
                and not parsed.fragment and request.headers.get('Sec-Fetch-Site', 'same-origin') == 'same-origin')
    except ValueError:
        return False


def _parts(filename, subfolder):
    if not isinstance(filename, str) or not isinstance(subfolder, str):
        raise InputError()
    if not filename or '/' in filename or '\\' in filename or filename in ('.', '..'):
        raise InputError()
    if os.path.splitext(filename)[1].lower() not in ALLOWED_EXT:
        raise InputError()
    if '\\' in subfolder or subfolder.startswith('/'):
        raise InputError()
    parts = subfolder.split('/') if subfolder else []
    if any(p in ('', '.', '..') for p in parts) or any('\x00' in p for p in parts + [filename]):
        raise InputError()
    return parts + [filename]


def _read_image(root, filename, subfolder):
    """Open every component without following symlinks; bound bytes read."""
    parts = _parts(filename, subfolder)
    fds = []
    try:
        current = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(current)
        for part in parts[:-1]:
            current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            fds.append(current)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        fds.append(fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= MAX_IMAGE:
            raise InputError()
        chunks, remaining = [], MAX_IMAGE + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b''.join(chunks)
        if not 0 < len(data) <= MAX_IMAGE:
            raise InputError()
        # Reject mislabeled non-image content without trusting filename alone.
        ext = os.path.splitext(filename)[1].lower()
        valid = (ext == '.png' and data.startswith(b'\x89PNG\r\n\x1a\n') or
                 ext in ('.jpg', '.jpeg') and data.startswith(b'\xff\xd8\xff') or
                 ext == '.webp' and data.startswith(b'RIFF') and data[8:12] == b'WEBP')
        if not valid:
            raise InputError()
        return data
    except (OSError, ValueError):
        raise InputError() from None
    finally:
        for fd in reversed(fds):
            os.close(fd)


async def _telegram_send(token, chat, filename, data):
    """Send original as a document; no automatic retries, redirects, or error logging."""
    try:
        form = FormData()
        form.add_field('chat_id', chat)
        form.add_field('document', data, filename=filename, content_type='application/octet-stream')
        async with ClientSession(timeout=ClientTimeout(total=120), trust_env=True) as session:
            async with session.post('https://api.telegram.org/bot' + token + '/sendDocument',
                                    data=form, allow_redirects=False) as response:
                if response.status == 429:
                    return 'rate_limited'
                if response.status >= 500 or 300 <= response.status < 400:
                    return 'unknown'
                if response.status >= 400:
                    return 'failed'
                try:
                    payload = await response.json()
                except Exception:
                    return 'unknown'
                return 'sent' if payload.get('ok') is True else 'failed'
    except Exception:
        return 'unknown'


class ImageContextSender:
    def __init__(self, server, output_directory, multi_user=False, send=None):
        self.server = server
        self.output_directory = output_directory
        self.multi_user = multi_user
        self.send = send or _telegram_send
        self._jobs = {}
        self._semaphore = asyncio.Semaphore(2)

    async def _run(self, token, chat, filename, data):
        async with self._semaphore:
            try:
                result = await self.send(token, chat, filename, data)
                return result if result in ('sent', 'failed', 'rate_limited', 'unknown') else 'unknown'
            except Exception:
                return 'unknown'

    async def preview(self, request):
        if not _same_origin(request):
            return web.json_response({'status': 'forbidden'}, status=403)
        if self.multi_user:
            return web.json_response({'status': 'multi_user_unsupported'}, status=403)
        try:
            if request.content_type != 'application/json' or (request.content_length is not None and request.content_length > 8192):
                raise InputError()
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {'filename', 'subfolder', 'type'} or body['type'] != 'output':
                raise InputError()
            data = await asyncio.to_thread(_read_image, self.output_directory(), body['filename'], body['subfolder'])
            ext = os.path.splitext(body['filename'])[1].lower()
            content_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}[ext]
            return web.Response(body=data, content_type=content_type, headers={
                'Cache-Control': 'no-store', 'X-Image-SHA256': hashlib.sha256(data).hexdigest(),
                'X-Content-Type-Options': 'nosniff'})
        except Exception:
            return web.json_response({'status': 'invalid_image'}, status=400, headers={'Cache-Control': 'no-store'})

    async def handle(self, request):
        def reply(status, http=200):
            return web.json_response({'status': status}, status=http, headers={'Cache-Control': 'no-store'})

        if not _same_origin(request):
            return reply('forbidden', 403)
        # Multi-user headers do not authenticate users. Do not enable this mode.
        if self.multi_user:
            return reply('multi_user_unsupported', 403)
        if request.content_type != 'application/json' or (request.content_length is not None and request.content_length > 8192):
            return reply('invalid_request', 400)
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {'filename', 'subfolder', 'type', 'idempotency_key', 'sha256'} or body['type'] != 'output':
                raise ValueError()
            if not isinstance(body['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', body['sha256']):
                raise ValueError()
            key = body['idempotency_key']
            if not isinstance(key, str) or str(uuid.UUID(key)) != key:
                raise ValueError()
            _parts(body['filename'], body['subfolder'])
        except Exception:
            return reply('invalid_request', 400)
        identity = (body['filename'], body['subfolder'], body['type'], body['sha256'])
        if key in self._jobs:
            old_identity, task = self._jobs[key]
            if old_identity != identity:
                return reply('key_conflict', 409)
            return reply(await asyncio.shield(task))
        if len(self._jobs) >= MAX_KEYS:
            return reply('capacity_exhausted', 503)
        if sum(not task.done() for _, task in self._jobs.values()) >= 2:
            return reply('busy', 503)
        try:
            settings = self.server.user_manager.settings.get_settings(request)
            token, chat = settings.get('Telegram.BotToken'), settings.get('Telegram.DefaultChatId')
            if not isinstance(token, str) or not token.strip() or not isinstance(chat, (str, int)) or not str(chat).strip():
                return reply('missing_settings', 400)
            token, chat = token.strip(), str(chat).strip()
            # Avoid malformed URL components and all server-supplied destinations.
            if not re.fullmatch(r'[0-9]+:[A-Za-z0-9_-]+', token):
                return reply('missing_settings', 400)
            data = await asyncio.to_thread(_read_image, self.output_directory(), body['filename'], body['subfolder'])
            if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), body['sha256']):
                return reply('image_changed', 409)
        except InputError:
            return reply('invalid_image', 400)
        except web.HTTPUnauthorized:
            return reply('unauthorized', 401)
        except Exception:
            return reply('settings_unavailable', 503)
        # Recheck after the file read yielded to the event loop.
        if key in self._jobs:
            old_identity, task = self._jobs[key]
            if old_identity != identity:
                return reply('key_conflict', 409)
            return reply(await asyncio.shield(task))
        if len(self._jobs) >= MAX_KEYS:
            return reply('capacity_exhausted', 503)
        if sum(not task.done() for _, task in self._jobs.values()) >= 2:
            return reply('busy', 503)
        task = asyncio.create_task(self._run(token, chat, body['filename'], data))
        self._jobs[key] = (identity, task)
        return reply(await asyncio.shield(task))


def register_routes(server):
    """Call once during plugin initialization, before ComfyUI freezes its routes."""
    import folder_paths
    from comfy.cli_args import args
    sender = ImageContextSender(server, folder_paths.get_output_directory, multi_user=args.multi_user)
    server.routes.post('/telegram_sender/image_context/preview')(sender.preview)
    server.routes.post('/telegram_sender/image_context')(sender.handle)
    return sender
