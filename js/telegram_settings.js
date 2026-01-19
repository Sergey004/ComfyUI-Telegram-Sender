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
        tooltip: "Telegram bot token from @BotFather",
    },
    {
        id: "Telegram.DefaultChatId",
        name: "Default Chat ID",
        type: "string",
        defaultValue: "",
        tooltip: "Default chat/channel ID for sending images",
    },
    // --- НАЧАЛО ИЗМЕНЕНИЙ: КАСТОМНОЕ ПОЛЕ LORA MAPPING ---
    {
        id: "Telegram.LoraMapping",
        name: "LoRA to Channel Mapping",
        // Вместо строки используем функцию для создания HTML элемента
        type: (name, setter, value) => {
            const input = document.createElement("textarea");
            
            // Стилизуем под ComfyUI (темная тема)
            input.value = value || "";
            input.style.width = "100%";     // На всю ширину
            input.style.height = "120px";   // Высота 120 пикселей
            input.style.resize = "vertical";// Можно растягивать вниз
            input.style.borderRadius = "4px";
            input.style.backgroundColor = "var(--comfy-input-bg)"; // Цвет фона Comfy
            input.style.color = "var(--input-text)";               // Цвет текста Comfy
            input.style.border = "1px solid var(--border-color)";
            input.style.padding = "5px";
            input.style.fontFamily = "monospace"; // Моноширинный шрифт для удобства
            
            // Сохраняем значение при изменении (когда убрали фокус)
            input.addEventListener("change", () => {
                setter(input.value);
            });
            
            return input;
        },
        defaultValue: "",
        tooltip: "Format (one per line):\nlora_name:chat_id\nanime:-100123456",
    },
    // --- КОНЕЦ ИЗМЕНЕНИЙ ---
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
];

app.registerExtension({
    name: "comfy.telegram_sender",
    settings: TELEGRAM_SETTINGS,
    
    // Эта функция запускается один раз при загрузке страницы ComfyUI
    async setup() {
        try {
            // 1. Проверяем, пуст ли токен в ТЕКУЩИХ настройках ComfyUI
            // Если там что-то есть, значит пользователь уже настроил или миграция прошла
            const currentToken = app.ui.settings.getSettingValue("Telegram.BotToken", "");
            
            if (currentToken) {
                return; // Миграция не нужна, выходим
            }

            console.log("[Telegram Sender] 📥 Checking for legacy config...");

            // 2. Запрашиваем данные у нашего Python API (который мы добавили на Шаге 1)
            const response = await api.fetchApi("/telegram_sender/get_legacy_config");
            
            if (response.status === 200) {
                const data = await response.json();
                
                // Проверяем, пришло ли что-то полезное
                if (data && (data.bot_token || data.default_chat_id)) {
                    console.log("[Telegram Sender] ♻️ Legacy config found! Migrating settings safely via UI API...");
                    
                    // 3. Используем официальный API ComfyUI для установки значений
                    // Это инициирует правильное сохранение файла comfy.settings.json самим ComfyUI
                    
                    if (data.bot_token) {
                        app.ui.settings.setSettingValue("Telegram.BotToken", data.bot_token);
                    }
                    if (data.default_chat_id) {
                        app.ui.settings.setSettingValue("Telegram.DefaultChatId", data.default_chat_id);
                    }
                    if (data.lora_mapping) {
                        app.ui.settings.setSettingValue("Telegram.LoraMapping", data.lora_mapping);
                    }
                    if (data.nsfw_channel_id) {
                        app.ui.settings.setSettingValue("Telegram.NSFWChannelId", data.nsfw_channel_id);
                    }
                    if (data.unsorted_channel_id) {
                        app.ui.settings.setSettingValue("Telegram.UnsortedChannelId", data.unsorted_channel_id);
                    }
                    
                    console.log("[Telegram Sender] ✅ Settings migrated successfully!");
                    app.extensionManager.toast.add({
                    severity: 'success',
                    summary: '✅ Settings migrated successfully!',
                    detail: 'Telegram settings migrated successfully!',
                    life: 3000
                    });
                }
            }
        } catch (error) {
            console.error("[Telegram Sender] Migration check failed:", error);
        }
    }
});