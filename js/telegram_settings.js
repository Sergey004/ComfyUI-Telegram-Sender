import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TELEGRAM_SETTINGS = [
    {
        id: "Telegram.About",
        name: "Telegram Sender",
        type: () => {return document.createElement('span')},
    },
    {
        id: "Telegram.BotToken",
        name: "Bot Token",
        type: "string",
        defaultValue: "",
        tooltip: "Telegram bot token from @BotFather\n\nFormat: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz\n\n💡 How to get:\n1. Open @BotFather in Telegram\n2. Send /newbot command\n3. Follow instructions\n4. Copy the token\n\n⚠️ SECURITY WARNING:\n- Never share your bot token\n- Token gives full control over your bot\n- Keep it secret!\n\n📖 Docs: https://core.telegram.org/bots/api",
        onChange: (value) => {
            // Save to Python backend
            saveTelegramSettings();
        }
    },
    {
        id: "Telegram.DefaultChatId",
        name: "Default Chat ID",
        type: "string",
        defaultValue: "",
        tooltip: "Default chat/channel ID for sending images\n\nPersonal chat: 123456789\nChannel: -1001234567890\nGroup: -1009876543210\n\n💡 Leave empty to specify in each node\n\n📖 How to get ID:\n1. Send message to bot\n2. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates\n3. Find \"chat\":{\"id\":NUMBER}\n\n⚠️ For channels: bot must be admin",
        onChange: (value) => {
            saveTelegramSettings();
        }
    },
    {
        id: "Telegram.LoraMapping",
        name: "LoRA to Channel Mapping",
        type: "text",
        defaultValue: "",
        tooltip: "Automatic routing based on LoRA names\n\nFormat: one per line: lora_name:chat_id\n\nExample:\nanime:-1001111111111\nrealistic:-1002222222222\ncharacter:-1003333333333\n\n💡 Partial match:\n- \"anime_style_v2\" matches \"anime\"\n- \"realistic_vision_xl\" matches \"realistic\"\n\n📖 See: Automatic LoRA Routing section in README",
        onChange: (value) => {
            saveTelegramSettings();
        }
    },
    {
        id: "Telegram.NSFWChannelId",
        name: "NSFW Channel ID",
        type: "string",
        defaultValue: "",
        tooltip: "Channel ID for NSFW content\n\nFormat: -1001234567890 (for channels)\n\n💡 Used when:\n- enable_nsfw_detection is enabled in node\n- \"nsfw\" keyword found in positive prompt\n\n⚠️ Bot must be admin in this channel",
        onChange: (value) => {
            saveTelegramSettings();
        }
    },
    {
        id: "Telegram.UnsortedChannelId",
        name: "Unsorted Channel ID",
        type: "string",
        defaultValue: "",
        tooltip: "Fallback channel for unrouted images\n\nFormat: -1001234567890 (for channels)\n\n💡 Used when:\n- No explicit chat_id specified\n- No LoRA routing match\n- No NSFW match\n\n📖 Channel determination priority:\n1. Explicit chat_id in node\n2. NSFW detection\n3. LoRA routing\n4. Default chat_id\n5. Unsorted channel",
        onChange: (value) => {
            saveTelegramSettings();
        }
    },
];

// Migrate settings from old config file if exists
async function migrateFromOldConfig() {
    try {
        const response = await api.fetchApi("/telegram/migrate_config", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.migrated) {
                console.log("[Telegram Sender] ✅ Settings migrated from old config");
                
                // Update UI settings with migrated values
                if (data.config) {
                    app.ui.settings.setSettingValue("Telegram.BotToken", data.config.bot_token || "");
                    app.ui.settings.setSettingValue("Telegram.DefaultChatId", data.config.default_chat_id || "");
                    app.ui.settings.setSettingValue("Telegram.LoraMapping", data.config.lora_mapping || "");
                    app.ui.settings.setSettingValue("Telegram.NSFWChannelId", data.config.nsfw_channel_id || "");
                    app.ui.settings.setSettingValue("Telegram.UnsortedChannelId", data.config.unsorted_channel_id || "");
                }
            }
        }
    } catch (error) {
        console.log("[Telegram Sender] ⚠️ Migration check skipped:", error);
    }
}

// Save settings to Python backend
async function saveTelegramSettings() {
    const config = {
        bot_token: app.ui.settings.getSettingValue("Telegram.BotToken") || "",
        default_chat_id: app.ui.settings.getSettingValue("Telegram.DefaultChatId") || "",
        lora_mapping: app.ui.settings.getSettingValue("Telegram.LoraMapping") || "",
        nsfw_channel_id: app.ui.settings.getSettingValue("Telegram.NSFWChannelId") || "",
        unsorted_channel_id: app.ui.settings.getSettingValue("Telegram.UnsortedChannelId") || ""
    };
    
    try {
        await api.fetchApi("/telegram/save_settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });
        console.log("[Telegram Sender] ✅ Settings saved to backend");
        
        // Show success toast notification
        app.extensionManager.toast.add({
            severity: 'success',
            summary: '✅ Settings Saved',
            detail: 'Telegram settings have been saved successfully!',
            life: 3000
        });
    } catch (error) {
        console.error("[Telegram Sender] ❌ Failed to save settings:", error);
        
        // Show error toast notification
        app.extensionManager.toast.add({
            severity: 'error',
            summary: '❌ Save Failed',
            detail: 'Failed to save settings. Check console for details.',
            life: 5000
        });
    }
}

app.registerExtension({
    name: "comfy.telegram_sender",
    settings: TELEGRAM_SETTINGS,
    
    async setup() {
        // Migrate settings from old config on startup
        await migrateFromOldConfig();
    }
});
