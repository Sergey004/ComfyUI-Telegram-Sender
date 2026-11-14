import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class InitIsolationTests(unittest.TestCase):
    def test_import_and_registration_failures_are_redacted(self):
        for mode in ('import', 'register'):
            with self.subTest(mode=mode):
                package = 'isolated_telegram_' + mode
                sender = types.ModuleType(package + '.telegram_sender')
                sender.NODE_CLASS_MAPPINGS = {'old': object()}
                sender.NODE_DISPLAY_NAME_MAPPINGS = {'old': 'Old'}
                server = types.ModuleType('server')
                server.PromptServer = types.SimpleNamespace(instance=object())
                context = types.ModuleType(package + '.telegram_image_context')
                def register(instance):
                    raise RuntimeError('SECRET_TOKEN https://secret.example')
                context.register_routes = register
                modules = {package + '.telegram_sender': sender, 'server': server}
                if mode == 'register':
                    modules[package + '.telegram_image_context'] = context
                # Prevent the unrelated legacy background fetch from running.
                civitai = types.ModuleType(package + '.telegram_modules.utils.civitai_fetch')
                civitai.fetch_missing = lambda **kwargs: None
                modules[civitai.__name__] = civitai
                spec = importlib.util.spec_from_file_location(package, ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
                module = importlib.util.module_from_spec(spec)
                output = io.StringIO()
                with patch.dict(sys.modules, modules), patch('threading.Thread') as thread, contextlib.redirect_stdout(output):
                    if mode == 'import':
                        # Explicitly fail the new-feature import with sensitive text.
                        import builtins
                        original = builtins.__import__
                        def injected(name, *args, **kwargs):
                            if name == 'telegram_image_context':
                                raise ImportError('SECRET_TOKEN https://secret.example')
                            return original(name, *args, **kwargs)
                        with patch('builtins.__import__', side_effect=injected):
                            spec.loader.exec_module(module)
                    else:
                        spec.loader.exec_module(module)
                    thread.assert_called_once()
                self.assertIs(module.NODE_CLASS_MAPPINGS, sender.NODE_CLASS_MAPPINGS)
                self.assertIs(module.NODE_DISPLAY_NAME_MAPPINGS, sender.NODE_DISPLAY_NAME_MAPPINGS)
                self.assertEqual(module.WEB_DIRECTORY, './js')
                self.assertIn('image context routes unavailable', output.getvalue())
                self.assertNotIn('SECRET_TOKEN', output.getvalue())
                self.assertNotIn('secret.example', output.getvalue())


if __name__ == '__main__':
    unittest.main()
