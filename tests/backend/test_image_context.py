import asyncio
import base64
import hashlib
import importlib.util
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web

spec = importlib.util.spec_from_file_location('image_context', Path(__file__).resolve().parents[2] / 'telegram_image_context.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Settings:
    def __init__(self):
        self.values = {'Telegram.BotToken': '123456:fake_token', 'Telegram.DefaultChatId': '-12345', 'Telegram.NSFWChannelId': '-999'}
        self.requests = []

    def get_settings(self, request):
        self.requests.append(request)
        return dict(self.values)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'a.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/aXcAAAAASUVORK5CYII='))
        self.settings = Settings()
        self.calls = []
        self.result = 'sent'
        async def send(*args):
            self.calls.append(args)
            await asyncio.sleep(.01)
            return self.result
        self.sender = mod.ImageContextSender(SimpleNamespace(user_manager=SimpleNamespace(settings=self.settings)), lambda: str(self.root), send=send)
        app = web.Application()
        app.router.add_post('/telegram_sender/image_context/preview', self.sender.preview)
        app.router.add_post('/telegram_sender/image_context', self.sender.handle)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.origin = str(self.client.make_url('/')).rstrip('/')
        self.body = {'filename': 'a.png', 'subfolder': '', 'type': 'output', 'idempotency_key': str(uuid.uuid4()), 'sha256': hashlib.sha256((self.root / 'a.png').read_bytes()).hexdigest()}

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def post(self, body=None, origin=None, **kwargs):
        response = await self.client.post('/telegram_sender/image_context', json=body if body is not None else self.body,
                                          headers={'Origin': self.origin if origin is None else origin, **kwargs})
        return response.status, await response.json()

    async def test_default_settings_and_no_secret_response(self):
        status, result = await self.post()
        self.assertEqual((status, result['status']), (200, 'sent'))
        self.assertEqual(self.calls[0][:3], ('123456:fake_token', '-12345', 'a.png'))
        self.assertNotIn('fake_token', json.dumps(result))
        self.assertEqual(len(self.settings.requests), 1)
        self.settings.values['Telegram.DefaultChatId'] = '-54321'
        self.body['idempotency_key'] = str(uuid.uuid4())
        await self.post()
        self.assertEqual(self.calls[-1][1], '-54321')

    async def test_missing_or_unavailable_settings(self):
        self.settings.values['Telegram.DefaultChatId'] = ''
        self.assertEqual((await self.post())[1]['status'], 'missing_settings')
        self.settings.get_settings = lambda request: (_ for _ in ()).throw(RuntimeError('fake_token'))
        status, result = await self.post()
        self.assertEqual((status, result['status']), (503, 'settings_unavailable'))
        self.assertNotIn('fake_token', str(result))
        self.assertFalse(self.calls)

    async def test_origin_and_multi_user_fail_closed(self):
        self.assertEqual((await self.post(origin='https://evil.example'))[0], 403)
        self.assertEqual((await self.post(origin=''))[0], 403)
        self.assertEqual((await self.post(**{'Sec-Fetch-Site': 'cross-site'}))[0], 403)
        self.sender.multi_user = True
        self.assertEqual((await self.post(**{'comfy-user': 'other'}))[1]['status'], 'multi_user_unsupported')
        self.assertFalse(self.settings.requests)

    async def test_reject_untrusted_fields_and_bad_paths(self):
        for changes in ({'chat_id': 'other'}, {'token': 'fake'}, {'filename': '../a.png'},
                        {'filename': '/a.png'}, {'subfolder': '../outside'}, {'type': 'temp'},
                        {'filename': 'a.txt'}, {'filename': 'missing.png'}, {'filename': ['a.png']}):
            with self.subTest(changes=changes):
                self.assertEqual((await self.post({**self.body, **changes}))[0], 400)
        (self.root / 'link.png').symlink_to(self.root / 'a.png')
        (self.root / 'dirlink').symlink_to(self.root, target_is_directory=True)
        for changes in ({'filename': 'link.png'}, {'subfolder': 'dirlink'},):
            self.assertEqual((await self.post({**self.body, **changes}))[1]['status'], 'invalid_image')
        (self.root / 'bad.png').write_bytes(b'not an image')
        self.assertEqual((await self.post({**self.body, 'filename': 'bad.png'}))[1]['status'], 'invalid_image')
        self.assertFalse(self.calls)

    async def test_dedup_concurrent_unknown_and_key_conflict(self):
        self.result = 'unknown'
        results = await asyncio.gather(self.post(), self.post(), self.post())
        self.assertEqual([x[1]['status'] for x in results], ['unknown'] * 3)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((await self.post())[1]['status'], 'unknown')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((await self.post({**self.body, 'filename': 'other.png'}))[1]['status'], 'key_conflict')

    async def test_preview_and_mutation_binding(self):
        image = {k: self.body[k] for k in ('filename', 'subfolder', 'type')}
        resp = await self.client.post('/telegram_sender/image_context/preview', json=image, headers={'Origin': self.origin})
        original = await resp.read()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers['X-Image-SHA256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(original, (self.root / 'a.png').read_bytes())
        (self.root / 'a.png').write_bytes(original + b'mutation')
        self.assertEqual((await self.post())[1]['status'], 'image_changed')
        self.assertFalse(self.calls)
        (self.root / 'replacement.png').write_bytes(original + b'replacement')
        (self.root / 'replacement.png').replace(self.root / 'a.png')
        self.assertEqual((await self.post())[1]['status'], 'image_changed')
        self.assertFalse(self.calls)
        (self.root / 'a.png').unlink()
        (self.root / 'a.png').symlink_to(self.root / 'other.png')
        (self.root / 'other.png').write_bytes(original)
        self.assertEqual((await self.post())[1]['status'], 'invalid_image')
        self.assertFalse(self.calls)

    async def test_send_uses_validated_bytes_after_file_changes(self):
        original = (self.root / 'a.png').read_bytes()
        async def send(token, chat, filename, data):
            (self.root / 'a.png').write_bytes(original + b'changed')
            self.calls.append((token, chat, filename, data))
            return 'sent'
        self.sender.send = send
        self.assertEqual((await self.post())[1]['status'], 'sent')
        self.assertEqual(self.calls[0][3], original)
        self.assertEqual((await self.post({**self.body, 'sha256': '0' * 64}))[1]['status'], 'key_conflict')

    async def test_rate_limit_is_not_retried(self):
        self.result = 'rate_limited'
        self.assertEqual((await self.post())[1]['status'], 'rate_limited')
        self.assertEqual((await self.post())[1]['status'], 'rate_limited')
        self.assertEqual(len(self.calls), 1)

    async def test_size_limit(self):
        (self.root / 'big.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'x' * mod.MAX_IMAGE)
        self.assertEqual((await self.post({**self.body, 'filename': 'big.png'}))[1]['status'], 'invalid_image')
        image = {'filename': 'big.png', 'subfolder': '', 'type': 'output'}
        response = await self.client.post('/telegram_sender/image_context/preview', json=image, headers={'Origin': self.origin})
        self.assertEqual(response.status, 400)

    async def test_chromium_real_http_end_to_end(self):
        import html
        import shutil
        import re
        browser = shutil.which('chromium')
        if not browser:
            self.skipTest('Chromium unavailable')
        script = (Path(__file__).resolve().parents[2] / 'js/telegram_image_context.js').read_text().split('\n', 2)[2]
        page = """<body><script>
        let extension;
        const img={filename:'a.png',subfolder:'',type:'output'};
        const app={nodeOutputs:{7:{images:[img]}},registerExtension:x=>extension=x};
        const api={apiURL:p=>p,fetchApi:(p,o)=>fetch(p,o)};
        """ + script + """
        (async()=>{
          const node={id:7,imgs:[{src:location.origin+'/view?filename=a.png&type=output'}]};
          if (isSecureContext || typeof crypto.randomUUID !== 'undefined' || typeof crypto.getRandomValues !== 'function') {
            document.body.dataset.result='unexpected security context'; return;
          }
          extension.getNodeMenuItems(node)[0].callback();
          for(let i=0;i<200 && document.querySelector('dialog button').disabled;i++) await new Promise(r=>setTimeout(r,10));
          document.querySelector('dialog button').click();
          for(let i=0;i<200;i++){
            await new Promise(r=>setTimeout(r,10));
            const status=document.querySelector('[role=status]').textContent;
            if(status !== '发送中…'){document.body.dataset.result=status;return;}
          }
          document.body.dataset.result='timeout';
        })();</script></body>"""
        # Separate application so its route can be added before router freeze.
        app = web.Application()
        app.router.add_get('/', lambda request: web.Response(text=page, content_type='text/html'))
        app.router.add_post('/telegram_sender/image_context/preview', self.sender.preview)
        app.router.add_post('/telegram_sender/image_context', self.sender.handle)
        server = TestServer(app)
        await server.start_server()
        try:
            url = f'http://insecure.test:{server.port}/'
            proc = await asyncio.create_subprocess_exec(browser, '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--no-proxy-server', '--host-resolver-rules=MAP insecure.test 127.0.0.1', '--virtual-time-budget=5000', '--dump-dom', url, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(proc.communicate(), 30)
            self.assertEqual(proc.returncode, 0)
            match = re.search(r'data-result="([^"]+)"', stdout.decode())
            self.assertIsNotNone(match, stdout.decode()[:1000])
            self.assertEqual(html.unescape(match.group(1)), '发送成功。')
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.calls[0][:3], ('123456:fake_token', '-12345', 'a.png'))
        finally:
            await server.close()


if __name__ == '__main__':
    unittest.main()
