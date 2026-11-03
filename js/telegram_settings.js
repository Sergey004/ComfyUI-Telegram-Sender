import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function requestScan(action, force = false) {
    try {
        const query = force ? "?force=1" : "";
        const response = await api.fetchApi(`/telegram_sender/civitai_fetch${action}${query}`, {
            method: "POST",
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }
        app.extensionManager.toast.add({
            severity: "info",
            summary: `CivitAI scan: ${data.state}`,
            detail: data.error || "Check the ComfyUI console for scan progress.",
            life: 5000,
        });
    } catch (error) {
        console.error("[Telegram Sender] CivitAI scan request failed:", error);
        app.extensionManager.toast.add({
            severity: "error",
            summary: "CivitAI scan request failed",
            detail: String(error),
            life: 5000,
        });
    }
}

function scanControl(name, action, force = false) {
    return () => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = name;
        button.addEventListener("click", () => requestScan(action, force));
        return button;
    };
}

const TELEGRAM_SETTINGS = [
    {
        id: "Telegram.About",
        name: "Telegram Sender",
        type: () => document.createElement("span"),
    },
    {
        id: "Telegram.BotToken",
        name: "Bot Token",
        type: "string",
        defaultValue: "",
        tooltip: "Telegram bot token from @BotFather",
    },
    {
        id: "Telegram.DefaultChatId",
        name: "Default Chat ID",
        type: "string",
        defaultValue: "",
        tooltip: "Default chat/channel ID for sending images",
    },
    {
        id: "Telegram.LoraMapping",
        name: "LoRA to Channel Mapping",
        type: (name, setter, value) => {
            const input = document.createElement("textarea");
            input.value = value || "";
            input.style.width = "100%";
            input.style.height = "120px";
            input.style.resize = "vertical";
            input.style.borderRadius = "4px";
            input.style.backgroundColor = "var(--comfy-input-bg)";
            input.style.color = "var(--input-text)";
            input.style.border = "1px solid var(--border-color)";
            input.style.padding = "5px";
            input.style.fontFamily = "monospace";
            input.addEventListener("change", () => setter(input.value));
            return input;
        },
        defaultValue: "",
        tooltip: "Format (one per line):\nlora_name:chat_id\nanime:-100123456",
    },
    {
        id: "Telegram.NSFWChannelId",
        name: "NSFW Channel ID",
        type: "string",
        defaultValue: "",
        tooltip: "Channel ID for NSFW content",
    },
    {
        id: "Telegram.UnsortedChannelId",
        name: "Unsorted Channel ID",
        type: "string",
        defaultValue: "",
        tooltip: "Fallback channel for unrouted images",
    },
    {
        id: "Telegram.CivitaiApiKey",
        name: "CivitAI API Key",
        type: "string",
        defaultValue: "",
        tooltip: "Your CivitAI API key for downloading gated/private models and NSFW previews.",
    },
    {
        id: "Telegram.CivitaiNsfwLevel",
        name: "CivitAI Max NSFW Level",
        type: "combo",
        options: ["PG", "PG-13", "R", "X", "XXX"],
        defaultValue: "PG",
        tooltip: "Maximum NSFW level for CivitAI preview images.",
    },
    {
        id: "Telegram.AutoCivitaiFetch",
        name: "Automatically fetch CivitAI model information on startup",
        type: "boolean",
        defaultValue: true,
        tooltip: "Start one background scan after the ComfyUI frontend initializes.",
    },
    {
        id: "Telegram.CivitaiFetchNow",
        name: "Manual CivitAI scan",
        type: scanControl("Scan now", "", true),
        defaultValue: "",
    },
    {
        id: "Telegram.CivitaiFetchStop",
        name: "Cancel CivitAI scan",
        type: scanControl("Cancel scan", "/cancel"),
        defaultValue: "",
    },
];

app.registerExtension({
    name: "comfy.telegram_sender",
    settings: TELEGRAM_SETTINGS,

    async setup() {
        try {
            const currentToken = app.ui.settings.getSettingValue("Telegram.BotToken", "");
            if (!currentToken) {
                const response = await api.fetchApi("/telegram_sender/get_legacy_config");
                if (!response.ok) {
                    throw new Error(`Legacy config request failed: HTTP ${response.status}`);
                }
                const data = await response.json();
                if (data && (data.bot_token || data.default_chat_id)) {
                    const settingMap = {
                        bot_token: "Telegram.BotToken",
                        default_chat_id: "Telegram.DefaultChatId",
                        lora_mapping: "Telegram.LoraMapping",
                        nsfw_channel_id: "Telegram.NSFWChannelId",
                        unsorted_channel_id: "Telegram.UnsortedChannelId",
                        civitai_api_key: "Telegram.CivitaiApiKey",
                        civitai_nsfw_level: "Telegram.CivitaiNsfwLevel",
                    };
                    for (const [source, setting] of Object.entries(settingMap)) {
                        if (data[source]) {
                            app.ui.settings.setSettingValue(setting, data[source]);
                        }
                    }
                    app.extensionManager.toast.add({
                        severity: "success",
                        summary: "Settings migrated successfully",
                        detail: "Telegram settings migrated successfully.",
                        life: 3000,
                    });
                }
            }
        } catch (error) {
            console.error("[Telegram Sender] Migration check failed:", error);
        }

        if (app.ui.settings.getSettingValue("Telegram.AutoCivitaiFetch", true)) {
            await requestScan("");
        }
    },
});
