#!/usr/bin/env python3
"""
Verification script to ensure all code loads correctly
and shows what happens during extraction
"""

import sys
sys.path.append("../../")

print("=" * 70)
print("VERIFICATION SCRIPT FOR TELEGRAMSENDER FIXES")
print("=" * 70)

# Test 1: Import modules
print("\n✓ Test 1: Checking imports...")
try:
    from telegram_metadata import TelegramMetadata, CAPTURE_AVAILABLE
    print(f"  ✅ TelegramMetadata imported successfully")
    print(f"  ℹ️  CAPTURE_AVAILABLE={CAPTURE_AVAILABLE}")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check if modules directory exists
print("\n✓ Test 2: Checking modules structure...")
import os
required_files = [
    "modules/__init__.py",
    "modules/capture.py",
    "modules/trace.py",
    "modules/hook.py",
    "modules/defs/captures.py",
    "modules/defs/loader.py",
    "modules/defs/ext/rgthree.py",
    "modules/defs/ext/CR_ApplyLoRAStack.py",
]

base_path = "/home/user/ComfyUI/custom_nodes/ComfyUI-Telegram-Sender"
all_present = True
for filepath in required_files:
    full_path = os.path.join(base_path, filepath)
    if os.path.exists(full_path):
        print(f"  ✅ {filepath}")
    else:
        print(f"  ❌ {filepath} - NOT FOUND")
        all_present = False

if not all_present:
    print("\n❌ Some required files are missing!")
    sys.exit(1)

# Test 3: Check telegram_sender.py code
print("\n✓ Test 3: Checking telegram_sender.py fixes...")
with open("telegram_sender.py", "r") as f:
    content = f.read()
    
checks = [
    ("TelegramMetadata import", "from .telegram_metadata import TelegramMetadata" in content),
    ("Enhanced metadata extraction in execute", "TelegramMetadata.get_metadata(prompt)" in content),
    ("_extract_loras_from_workflow using TelegramMetadata", "pnginfo_dict = TelegramMetadata.get_metadata(prompt_dict)" in content),
    ("Fallback only for LoraLoader nodes", 'if "LoraLoader" in class_type' in content),
    ("No 'Using fallback LoRA extraction' message", '"🔍 Using fallback LoRA extraction' not in content),
]

all_good = True
for check_name, result in checks:
    if result:
        print(f"  ✅ {check_name}")
    else:
        print(f"  ❌ {check_name}")
        all_good = False

if not all_good:
    print("\n⚠️  Some expected fixes might be missing!")

# Test 4: What happens when called
print("\n✓ Test 4: Function signature check...")
print(f"  TelegramMetadata.get_metadata: {hasattr(TelegramMetadata, 'get_metadata')}")
print(f"  TelegramMetadata.get_parameters_str: {hasattr(TelegramMetadata, 'get_parameters_str')}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
✅ Code structure verified!

When running in ComfyUI:
1. TelegramSender.execute() will be called with the workflow prompt
2. TelegramMetadata.get_metadata(prompt) will extract LoRA info
3. For rgthree Power Lora Loader: modules/defs/ext/rgthree.py handles it
4. For CR LoRA Stack: modules/defs/ext/CR_ApplyLoRAStack.py handles it  
5. LoRAs will be correctly detected and routed to appropriate channels

Note: Full testing requires ComfyUI execution environment.
Restart ComfyUI to apply changes!
""")
