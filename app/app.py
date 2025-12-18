import os
import json
import requests
import time
import threading
import uuid
import base64
import tempfile
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from notion_client import Client
from openai import OpenAI  # Using the new OpenAI API
import anthropic  # pip install anthropic
from app.pushover_client import send_pushover

# Application version updated to 1.6.0
VERSION = "1.6.0"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")
app.jinja_env.globals.update(enumerate=enumerate)

# Filter out noisy polling endpoints from Flask/Werkzeug logs
class QuietPollingFilter(logging.Filter):
    """Filter to suppress log messages for polling endpoints."""
    QUIET_ENDPOINTS = ['/logs', '/tag_logs', '/upload_logs', '/automation_logs', 
                       '/logs_comparator', '/logs_ai_comparator']
    
    def filter(self, record):
        message = record.getMessage()
        # Filter out GET requests to polling endpoints
        for endpoint in self.QUIET_ENDPOINTS:
            if f'GET {endpoint}' in message or f'"{endpoint} HTTP' in message:
                return False
        return True

# Apply filter to Werkzeug logger (Flask's HTTP request logger)
logging.getLogger('werkzeug').addFilter(QuietPollingFilter())

# Global state for automation
current_automation_state = None

# Configuration file paths
# New persistent location (Docker volume)
DATA_DIR = "/data"
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
# Legacy location (inside app directory - will be migrated)
LEGACY_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def migrate_config_if_needed():
    """
    Migrate config from legacy location (app/config.json) to new persistent location (/data/config.json).
    This ensures configurations survive Docker container rebuilds/updates.
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if migration is needed
    legacy_exists = os.path.exists(LEGACY_CONFIG_FILE)
    new_exists = os.path.exists(CONFIG_FILE)
    
    if legacy_exists and not new_exists:
        # Migrate from legacy to new location
        try:
            import shutil
            shutil.copy2(LEGACY_CONFIG_FILE, CONFIG_FILE)
            print(f"[CONFIG] ✅ Migrated config from {LEGACY_CONFIG_FILE} to {CONFIG_FILE}", flush=True)
            # Remove legacy file after successful migration
            os.remove(LEGACY_CONFIG_FILE)
            print(f"[CONFIG] 🗑️ Removed legacy config file", flush=True)
        except Exception as e:
            print(f"[CONFIG] ❌ Migration failed: {e}", flush=True)
    elif legacy_exists and new_exists:
        # Both exist - keep the newer one, remove legacy
        try:
            legacy_mtime = os.path.getmtime(LEGACY_CONFIG_FILE)
            new_mtime = os.path.getmtime(CONFIG_FILE)
            if legacy_mtime > new_mtime:
                # Legacy is newer, update the new location
                import shutil
                shutil.copy2(LEGACY_CONFIG_FILE, CONFIG_FILE)
                print(f"[CONFIG] ✅ Updated config from newer legacy file", flush=True)
            os.remove(LEGACY_CONFIG_FILE)
            print(f"[CONFIG] 🗑️ Removed legacy config file", flush=True)
        except Exception as e:
            print(f"[CONFIG] ⚠️ Could not clean up legacy config: {e}", flush=True)
    elif new_exists:
        print(f"[CONFIG] ✅ Using persistent config at {CONFIG_FILE}", flush=True)
    else:
        print(f"[CONFIG] ℹ️ No existing config found, will create new one at {CONFIG_FILE}", flush=True)

# Run migration on startup
migrate_config_if_needed()

DEFAULT_DESCRIPTION_PROMPT = (
    "You are a visual analyst who must describe with maximum precision the image shown to you, without omitting any details. "
    "Describe the image in a continuous and plain-text format, without structuring your response into sections or adding additional explanations. "
    "Detail the environment, each object, figure, text, or symbol, specifying colors, textures, shapes, shadows, relative positions, and any nuances of light or contrast. "
    "If people are present, indicate their physical characteristics, clothing, expressions, and postures. "
    "Your description must allow a blind person to mentally recreate the image in its entirety. "
    "Remember: do not offer any extra commentary, only the precise and complete description in a single block of text. "
    "Ignore text pages; focus exclusively on images or photographs."
)

# Límite de tokens por defecto: 500 (se usará este valor a menos que el usuario indique lo contrario)
DEFAULT_MAX_TOKENS = 500

# Global arrays for logs (to be displayed in the web interface)
processing_log = []
tag_processing_log = []
file_upload_processing_log = []
automation_processing_log = []

# Default model catalog shared across backend/frontend
DEFAULT_MODELS = [
    {"name": "GPT-4o", "model": "gpt-4o", "api": "OpenAI"},
    {"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api": "OpenAI"},
    {"name": "GPT-5", "model": "gpt-5", "api": "OpenAI"},
    {"name": "GPT-5.1", "model": "gpt-5.1", "api": "OpenAI"},
    {"name": "GPT-5 Mini", "model": "gpt-5-mini", "api": "OpenAI"},
    {"name": "GPT-5 Nano", "model": "gpt-5-nano", "api": "OpenAI"},
    {"name": "DeepSeek Chat", "model": "deepseek-chat", "api": "DeepSeek"},
    {"name": "DeepSeek Reasoner", "model": "deepseek-reasoner", "api": "DeepSeek"},
    {"name": "Gemini 2.0 Flash", "model": "gemini-2.0-flash", "api": "Gemini"},
    {"name": "Gemini 2.5 Flash", "model": "gemini-2.5-flash", "api": "Gemini"},
    {"name": "Gemini 2.5 Flash Lite", "model": "gemini-2.5-flash-lite", "api": "Gemini"},
    {"name": "Gemini 3 Pro Preview", "model": "gemini-3-pro-preview", "api": "Gemini"},
    {"name": "Gemini 3 Pro", "model": "gemini-3-pro", "api": "Gemini"},
    {"name": "Gemini 3 Flash Preview", "model": "gemini-3-flash-preview", "api": "Gemini"},
    {"name": "Gemini 3 Flash", "model": "gemini-3-flash", "api": "Gemini"},
    {"name": "Claude 3.5 Sonnet Latest", "model": "claude-3-5-sonnet-latest", "api": "Anthropic"},
    {"name": "Claude 3.5 Haiku Latest", "model": "claude-3-5-haiku-latest", "api": "Anthropic"},
    {"name": "Claude 3 Opus Latest", "model": "claude-3-opus-latest", "api": "Anthropic"},
    {"name": "o3 Mini", "model": "o3-mini", "api": "OpenAI", "is_reasoning": True, "reasoning_effort": "medium"},
    {"name": "o1 Mini", "model": "o1-mini", "api": "OpenAI", "is_reasoning": True, "reasoning_effort": None},
    {"name": "O1", "model": "o1", "api": "OpenAI", "is_reasoning": True, "reasoning_effort": None}
]


def get_default_models():
    """Return a deep copy of the default model catalog."""
    return [dict(model) for model in DEFAULT_MODELS]


def ensure_default_models(config):
    """Ensure mandatory models exist in config and keep list sorted."""
    models = config.setdefault("models_list", [])
    existing_models = {model.get("model"): model for model in models if model.get("model")}
    added = False
    for default_model in DEFAULT_MODELS:
        if default_model["model"] not in existing_models:
            models.append(dict(default_model))
            added = True
    if added:
        models.sort(key=lambda x: x["name"].lower())
    return added

def load_config():
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            "notion_db_id": "",
            "columns": {"image": "", "description": ""},
            "model": "",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "prompts": [],
            "default_description": {"prompt": "", "language": ""},
            "tag_config": {
                "allowed_tags": "",
                "tag_prompt": "",
                "tag_column": "",
                "notion_db_id": "",
                "max_tokens": DEFAULT_MAX_TOKENS,
                "max_tags": 1
            },
            "notion_db_ids": [],
            "models_list": get_default_models(),
            "columns_configs": [],
            "tag_configs": [],
            "comparator_configs": [],
            "output_configs": [],
            "file_upload_configs": [],
            "languages": ["Spanish", "English"]
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            config.setdefault("default_description", {"prompt": "", "language": ""})
            config.setdefault("tag_config", {
                "allowed_tags": "",
                "tag_prompt": "",
                "tag_column": "",
                "notion_db_id": "",
                "max_tokens": DEFAULT_MAX_TOKENS,
                "max_tags": 1
            })
            config.setdefault("notion_db_ids", [])
            models_added = ensure_default_models(config)
            modelo_api_map = {
                "GPT-4o": "OpenAI",
                "GPT-4o-mini": "OpenAI",
                "GPT-5": "OpenAI",
                "GPT-5.1": "OpenAI",
                "GPT-5 Mini": "OpenAI",
                "GPT-5 Nano": "OpenAI",
                "DeepSeek Chat": "DeepSeek",
                "DeepSeek Reasoner": "DeepSeek",
                "Gemini 2.0 Flash": "Gemini",
                "Gemini 2.5 Flash": "Gemini",
                "Gemini 2.5 Flash Lite": "Gemini",
                "Gemini 3 Pro Preview": "Gemini",
                "Gemini 3 Pro": "Gemini",
                "Gemini 3 Flash Preview": "Gemini",
                "Gemini 3 Flash": "Gemini",
                "Claude 3.5 Sonnet Latest": "Anthropic",
                "Claude 3.5 Haiku Latest": "Anthropic",
                "Claude 3 Opus Latest": "Anthropic",
                "o3 Mini": "OpenAI",
                "o1 Mini": "OpenAI",
                "O1": "OpenAI"
            }
            for mod in config.get("models_list", []):
                if "api" not in mod or not mod["api"]:
                    mod["api"] = modelo_api_map.get(mod["name"], "OpenAI")
                if mod["api"] == "OpenAI" and mod["model"] in ["o3-mini", "o1-mini", "o1"]:
                    mod.setdefault("is_reasoning", True)
                    if mod["model"] == "o3-mini":
                        mod.setdefault("reasoning_effort", "medium")
                    else:
                        mod.setdefault("reasoning_effort", None)
                else:
                    mod.setdefault("is_reasoning", False)
            config.setdefault("columns_configs", [])
            config.setdefault("tag_configs", [])
            config.setdefault("comparator_configs", [])
            config.setdefault("output_configs", [])
            config.setdefault("file_upload_configs", [])
            config.setdefault("languages", ["Spanish", "English"])
            for prompt in config.get("prompts", []):
                if "name" not in prompt:
                    prompt["name"] = prompt.get("prompt", "Unnamed Prompt")
                if "category" not in prompt:
                    prompt["category"] = "description"
            if models_added:
                save_config(config)
            return config
    except Exception as e:
        print(f"❌ Error loading config: {e}", flush=True)
        return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving config: {e}", flush=True)

# Helper function to get Pushover notification config without Flask context
def get_pushover_config_for_thread():
    """
    Get Pushover notification configuration for use in background threads.
    This function works outside Flask application context.
    """
    try:
        config = load_config()
        # Default config if not set
        default = {
            'start_process': False,
            'end_process': True,
            'start_automation': False,
            'end_automation': True
        }
        return config.get('pushover_notif_config', default)
    except Exception as e:
        print(f"❌ Error loading Pushover config: {e}", flush=True)
        return {
            'start_process': False,
            'end_process': True,
            'start_automation': False,
            'end_automation': True
        }

def is_valid_uuid(val):
    try:
        if len(val) == 32:
            val = f"{val[0:8]}-{val[8:12]}-{val[12:16]}-{val[16:20]}-{val[20:32]}"
        uuid.UUID(val)
        return True
    except ValueError:
        return False

# Endpoints to retrieve logs for the web interface
@app.route("/logs", methods=["GET"])
def get_logs():
    return jsonify({"log": "\n".join(processing_log)})

@app.route("/tag_logs", methods=["GET"])
def get_tag_logs():
    return jsonify({"log": "\n".join(tag_processing_log)})

# NEW: Comparator logs endpoint
@app.route("/logs_comparator", methods=["GET"])
def get_comparator_logs():
    return jsonify({"log": "\n".join(comparator_processing_log)})

# NEW: AI Comparator logs endpoint
ai_comparator_background_thread = None
stop_ai_comparator_processing = False
ai_comparator_processing_log = []

@app.route("/logs_ai_comparator", methods=["GET"])
def get_ai_comparator_logs():
    return jsonify({"log": "\n".join(ai_comparator_processing_log)})

# File upload logs endpoint
@app.route("/upload_logs", methods=["GET"])
def get_upload_logs():
    return jsonify({"log": "\n".join(file_upload_processing_log)})

# Global variables for background processes
background_thread = None
tag_background_thread = None
stop_processing = False
stop_tag_processing = False

# NEW: Global variables for comparator process
comparator_background_thread = None
stop_comparator_processing = False
comparator_processing_log = []

# API environment variables
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

notion = Client(auth=NOTION_API_KEY)


def query_database(database_id, **kwargs):
    """Query a Notion database with timeout. Uses direct API call for better control."""
    import time as time_module
    
    try:
        print(f"[DEBUG] query_database: Making API call...", flush=True)
        start = time_module.time()
        
        # Always use direct API call with timeout for better control
        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        response = requests.post(url, headers=headers, json=kwargs, timeout=30)
        response.raise_for_status()
        
        elapsed = time_module.time() - start
        print(f"[DEBUG] query_database: Response in {elapsed:.2f}s", flush=True)
        
        return response.json()
    except requests.exceptions.Timeout:
        error_message = f"❌ Timeout querying Notion database (>30s)"
        print(error_message, flush=True)
        processing_log.append(error_message)
        raise
    except Exception as e:
        error_message = f"❌ Error querying Notion database: {e}"
        print(error_message, flush=True)
        processing_log.append(error_message)
        raise

# --- OPENAI CLIENT ---
class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def analyze_image(self, image_url, prompt, model, token_limit):
        if image_url:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        config = load_config()
        is_reasoning = False
        reasoning_effort = None
        for mod in config.get("models_list", []):
            if mod["model"] == model and mod["api"] == "OpenAI":
                is_reasoning = mod.get("is_reasoning", False)
                reasoning_effort = mod.get("reasoning_effort")
                break

        try:
            if is_reasoning and reasoning_effort is not None:
                if token_limit is not None:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=token_limit,
                        reasoning_effort=reasoning_effort,
                        timeout=60
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=60
                    )
            else:
                if token_limit is not None:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=token_limit,
                        timeout=60
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=60
                    )
            msg = f"Response from {model} (image): {response.choices[0].message.content}"
            print(msg, flush=True)
            processing_log.append(msg)
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = f"❌ OpenAI API error: {e}"
            print(err_msg, flush=True)
            processing_log.append(err_msg)
            raise ValueError(err_msg)

    def analyze_tag(self, prompt, model, max_tokens):
        config = load_config()
        is_reasoning = False
        reasoning_effort = None
        for mod in config.get("models_list", []):
            if mod["model"] == model and mod["api"] == "OpenAI":
                is_reasoning = mod.get("is_reasoning", False)
                reasoning_effort = mod.get("reasoning_effort")
                break
        try:
            if is_reasoning and reasoning_effort is not None:
                if max_tokens is not None:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        timeout=60
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=60
                    )
            else:
                if max_tokens is not None:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        timeout=60
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=60
                    )
            msg = f"Response from {model} (tag): {response.choices[0].message.content}"
            print(msg, flush=True)
            tag_processing_log.append(msg)
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = f"❌ OpenAI API error: {e}"
            print(err_msg, flush=True)
            tag_processing_log.append(err_msg)
            raise ValueError(err_msg)

openai_api = OpenAIClient(OPENAI_API_KEY)

# --- DEEPSEEK CLIENT ---
class DeepSeekClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def analyze_image(self, image_url, prompt, model, token_limit):
        if image_url:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            if token_limit is not None:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=token_limit,
                    timeout=60
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=60
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"❌ DeepSeek API error: {e}")

    def analyze_tag(self, prompt, model, max_tokens):
        try:
            if max_tokens is not None:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    timeout=60
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"❌ DeepSeek API error: {e}")

deepseek_api = DeepSeekClient(DEEPSEEK_API_KEY)

# --- GEMINI CLIENT ---
class GeminiClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    def encode_image_from_url(self, image_url):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            raise ValueError(f"❌ Gemini error encoding image: {e}")

    def analyze_image(self, image_url, prompt, model, token_limit):
        base64_image = self.encode_image_from_url(image_url)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "auto"}}
            ]
        }]
        try:
            if token_limit is not None:
                # Add 10% margin to max_tokens for Gemini to avoid losing content
                # Gemini models may return slightly more tokens than requested
                effective_token_limit = int(token_limit * 1.1)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=effective_token_limit,
                    timeout=60
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=60
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"❌ Gemini API error: {e}")

    def analyze_tag(self, prompt, model, max_tokens):
        try:
            if max_tokens is not None:
                # Add 10% margin to max_tokens for Gemini to avoid losing content
                # Gemini models may return slightly more tokens than requested
                effective_max_tokens = int(max_tokens * 1.1)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=effective_max_tokens,
                    timeout=60
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"❌ Gemini API error: {e}")

gemini_api = None
if GEMINI_API_KEY:
    gemini_api = GeminiClient(GEMINI_API_KEY)

# --- ANTHROPIC CLIENT ---
class AnthropicClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_image(self, image_url, prompt, model, token_limit):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            raise ValueError(f"❌ Error fetching image: {e}")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": prompt}
        ]
        try:
            if token_limit is not None:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=token_limit,
                    messages=[{"role": "user", "content": content}]
                )
            else:
                message = self.client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": content}]
                )
            return message.content.strip()
        except Exception as e:
            raise ValueError(f"❌ Anthropic API error: {e}")

    def analyze_tag(self, prompt, model, max_tokens):
        try:
            if max_tokens is not None:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                message = self.client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
            result = message.content[0].text
            return result.strip()
        except Exception as e:
            raise ValueError(f"❌ Anthropic API error: {e}")

    def create_message_batch(self, requests_list):
        url = "https://api.anthropic.com/v1/messages/batches"
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {"requests": requests_list}
        try:
            r = requests.post(url, headers=headers, json=data, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise ValueError(f"❌ Anthropic Batch API error: {e}")

anthropic_api = None
if ANTHROPIC_API_KEY:
    anthropic_api = AnthropicClient(ANTHROPIC_API_KEY)

def update_notion_tag(page_id, tag_string, tag_column, max_tags=None):
    try:
        tags = [t.strip() for t in tag_string.split(',') if t.strip()]
        if max_tags is not None:
            tags = tags[:max_tags]
        tags = [t if len(t) <= 100 else t[:100] for t in tags]
        notion.pages.update(
            page_id=page_id,
            properties={
                tag_column: {"multi_select": [{"name": t} for t in tags]}
            }
        )
    except Exception as e:
        print(f"❌ Error in update_notion_tag: {e}")
        tag_processing_log.append(f"❌ Error in update_notion_tag: {e}")

def analyze_image_choice(api_choice, image_url, prompt, model, token_limit=500):
    try:
        if api_choice.lower() == "deepseek":
            return deepseek_api.analyze_image(image_url, prompt, model, token_limit)
        elif api_choice.lower() == "gemini":
            if gemini_api is None:
                raise ValueError("Gemini API key not configured")
            return gemini_api.analyze_image(image_url, prompt, model, token_limit)
        elif api_choice.lower() == "anthropic":
            if anthropic_api is None:
                raise ValueError("Anthropic API key not configured")
            return anthropic_api.analyze_image(image_url, prompt, model, token_limit)
        else:
            return openai_api.analyze_image(image_url, prompt, model, token_limit)
    except Exception as e:
        print(f"❌ Error in analyze_image_choice: {e}")
        processing_log.append(f"❌ Error in analyze_image_choice: {e}")

def analyze_tag_choice(api_choice, description_text, prompt, allowed_tags, model, max_tokens, max_tags):
    prompt = prompt.format(max_tags=max_tags)
    full_prompt = f"{prompt}\nAllowed tags: {allowed_tags}\nDescription:\n{description_text}"
    try:
        if api_choice.lower() == "deepseek":
            return deepseek_api.analyze_tag(full_prompt, model, max_tokens)
        elif api_choice.lower() == "gemini":
            if gemini_api is None:
                raise ValueError("Gemini API key not configured")
            return gemini_api.analyze_tag(full_prompt, model, max_tokens)
        elif api_choice.lower() == "anthropic":
            if anthropic_api is None:
                raise ValueError("Anthropic API key not configured")
            return anthropic_api.analyze_tag(full_prompt, model, max_tokens)
        else:
            return openai_api.analyze_tag(full_prompt, model, max_tokens)
    except Exception as e:
        print(f"❌ Error in analyze_tag_choice: {e}")
        tag_processing_log.append(f"❌ Error in analyze_tag_choice: {e}")

def get_database_entries(notion_db_id, image_column, description_column):
    try:
        response = query_database(
            notion_db_id,
            filter={
                "and": [
                    {"property": description_column, "rich_text": {"is_empty": True}},
                    {"property": image_column, "files": {"is_not_empty": True}}
                ]
            }
        )
        return response.get("results", [])
    except Exception as e:
        print(f"❌ Error in get_database_entries: {e}")
        processing_log.append(f"❌ Error in get_database_entries: {e}")

def get_image_url_from_entry(entry, image_column):
    try:
        files = entry["properties"].get(image_column, {}).get("files", [])
        if files and isinstance(files, list) and len(files) > 0:
            if "file" in files[0] and "url" in files[0]["file"]:
                return files[0]["file"]["url"]
            elif "external" in files[0] and "url" in files[0]["external"]:
                return files[0]["external"]["url"]
        return None
    except Exception as e:
        print(f"❌ Error in get_image_url_from_entry: {e}")
        processing_log.append(f"❌ Error in get_image_url_from_entry: {e}")

def get_image_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return url
    except requests.exceptions.RequestException as e:
        print(f"❌ Error in get_image_url: {e}")
        processing_log.append(f"❌ Error in get_image_url: {e}")

def analyze_image_main(api_choice, image_url, prompt, model, token_limit):
    effective_prompt = prompt.strip()  # No fallback to default prompt
    try:
        return analyze_image_choice(api_choice, image_url, effective_prompt, model, token_limit)
    except Exception as e:
        print(f"❌ Error in analyze_image_main: {e}")
        processing_log.append(f"❌ Error in analyze_image_main: {e}")

def analyze_tag_main(api_choice, description_text, prompt, allowed_tags, model, max_tokens, max_tags):
    try:
        return analyze_tag_choice(api_choice, description_text, prompt, allowed_tags, model, max_tokens, max_tags)
    except Exception as e:
        print(f"❌ Error in analyze_tag_main: {e}")
        tag_processing_log.append(f"❌ Error in analyze_tag_main: {e}")

def update_notion_description(page_id, description, description_column):
    try:
        notion.pages.update(
            page_id=page_id,
            properties={
                description_column: {"rich_text": [{"text": {"content": description}}]}
            }
        )
    except Exception as e:
        print(f"❌ Error in update_notion_description: {e}")
        processing_log.append(f"❌ Error in update_notion_description: {e}")

def process_entries_background(notion_db_id, image_column, description_column, prompt, api_choice, model, token_limit, language, batch_flag):
    global stop_processing
    processing_log.append("🚀 Starting description process...")
    entries = get_database_entries(notion_db_id, image_column, description_column)
    total = len(entries)
    processing_log.append(f"ℹ️ Found {total} entries to process.")
    count = 0
    for entry in entries:
        if stop_processing:
            processing_log.append("⏹️ Description process stopped by user.")
            break
        try:
            page_id = entry["id"]
            url = get_image_url_from_entry(entry, image_column)
            if not url:
                processing_log.append(f"⚠️ Entry {page_id} has no image URL. Skipping.")
                continue
            verified_url = get_image_url(url)
            if not verified_url:
                processing_log.append(f"⚠️ Invalid URL for entry {page_id}. Skipping.")
                continue
            effective_prompt = prompt
            if language and language.strip():
                effective_prompt += f"\nPlease generate the description in {language}."
            description = analyze_image_main(api_choice, verified_url, effective_prompt, model, token_limit)
            if not description:
                processing_log.append(f"❌ Failed to analyze image for entry {page_id}.")
                continue
            update_notion_description(page_id, description, description_column)
            count += 1
            processing_log.append(f"✅ Processed entry {page_id} ({count}/{total}).")
            time.sleep(1)
        except Exception as e:
            processing_log.append(f"❌ Error processing entry {entry.get('id')}: {e}")
    processing_log.append("🏁 Description processing completed.")

def get_entries_for_tagging(notion_db_id, description_column, tag_column):
    try:
        response = query_database(
            notion_db_id,
            filter={
                "and": [
                    {"property": description_column, "rich_text": {"is_not_empty": True}},
                    {"property": tag_column, "multi_select": {"is_empty": True}}
                ]
            }
        )
        return response.get("results", [])
    except Exception as e:
        tag_processing_log.append(f"❌ Error fetching tagging entries: {e}")
        return []

def process_tag_entries_background(notion_db_id, description_column, tag_prompt, allowed_tags, api_choice, model, max_tokens, tag_column, max_tags):
    global stop_tag_processing
    tag_processing_log.append("🚀 Starting tagging process...")
    # Warn if no allowed_tags specified (AI will generate random/free-form tags)
    if not allowed_tags or not allowed_tags.strip():
        tag_processing_log.append("⚠️ WARNING: No allowed tags specified. The AI will generate random/free-form tags.")
    entries = get_entries_for_tagging(notion_db_id, description_column, tag_column)
    total = len(entries)
    tag_processing_log.append(f"ℹ️ Found {total} entries to tag.")
    count = 0
    for entry in entries:
        if stop_tag_processing:
            tag_processing_log.append("⏹️ Tagging process stopped by user.")
            break
        try:
            page_id = entry["id"]
            description_rich = entry["properties"][description_column]["rich_text"]
            if not description_rich or len(description_rich) == 0:
                tag_processing_log.append(f"⚠️ Entry {page_id} has no description. Skipping.")
                continue
            desc_text = " ".join([item.get("text", {}).get("content", "") for item in description_rich])
            if not desc_text.strip():
                tag_processing_log.append(f"⚠️ Entry {page_id} has empty description text. Skipping.")
                continue
            predicted_tag_string = analyze_tag_main(api_choice, desc_text, tag_prompt, allowed_tags, model, max_tokens, max_tags)
            if not predicted_tag_string:
                tag_processing_log.append(f"❌ Failed to determine tag for entry {page_id}.")
                continue
            update_notion_tag(page_id, predicted_tag_string, tag_column, max_tags)
            count += 1
            tag_processing_log.append(f"✅ Tagged entry {page_id} ({count}/{total}).")
            time.sleep(1)
        except Exception as e:
            tag_processing_log.append(f"❌ Error tagging entry {entry.get('id')}: {e}")
    tag_processing_log.append("🏁 Tagging process completed.")

def repeated_process_entries(repeat_count, notion_db_id, image_column, description_column, prompt, api_choice, model, token_limit, language, batch_flag):
    global stop_processing
    try:
        from app.pushover_client import send_pushover
        pushover_cfg = get_pushover_config_for_thread()
        if pushover_cfg.get('start_process'):
            send_pushover(f"Description process started.", title="NotyVisualScan", priority=0)
    except Exception as e:
        processing_log.append(f"⚠️ Pushover notification failed: {e}")
        pushover_cfg = {}
    
    for i in range(repeat_count):
        if stop_processing:
            processing_log.append("⏹️ Description process stopped by user during repetitions.")
            break
        processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        try:
            process_entries_background(notion_db_id, image_column, description_column, prompt, api_choice, model, token_limit, language, batch_flag)
            processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")
        except Exception as e:
            error_msg = f"❌ Error in repetition {i+1}: {e}"
            processing_log.append(error_msg)
            try:
                from app.pushover_client import send_pushover
                send_pushover(error_msg, title="NotyVisualScan Error", priority=1)
            except:
                pass
            continue
    
    try:
        if pushover_cfg.get('end_process'):
            from app.pushover_client import send_pushover
            send_pushover(f"Description process completed. Total repetitions: {repeat_count}", title="NotyVisualScan", priority=0)
    except Exception as e:
        processing_log.append(f"⚠️ Pushover notification failed: {e}")

def repeated_process_tag_entries(repeat_count, notion_db_id, description_column, tag_prompt, allowed_tags, api_choice, model, max_tokens, tag_column, max_tags):
    global stop_tag_processing
    try:
        pushover_cfg = get_pushover_config_for_thread()
        if pushover_cfg.get('start_process'):
            from app.pushover_client import send_pushover
            send_pushover(f"Tagging process started.", title="NotyVisualScan", priority=0)
    except Exception as e:
        tag_processing_log.append(f"⚠️ Pushover notification failed: {e}")
        pushover_cfg = {}
    
    for i in range(repeat_count):
        if stop_tag_processing:
            tag_processing_log.append("⏹️ Tagging process stopped by user during repetitions.")
            break
        tag_processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        try:
            process_tag_entries_background(notion_db_id, description_column, tag_prompt, allowed_tags, api_choice, model, max_tokens, tag_column, max_tags)
            tag_processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")
        except Exception as e:
            error_msg = f"❌ Error in repetition {i+1}: {e}"
            tag_processing_log.append(error_msg)
            try:
                from app.pushover_client import send_pushover
                send_pushover(error_msg, title="NotyVisualScan Error", priority=1)
            except:
                pass
            continue
    
    try:
        if pushover_cfg.get('end_process'):
            from app.pushover_client import send_pushover
            send_pushover(f"Tagging process completed. Total repetitions: {repeat_count}", title="NotyVisualScan", priority=0)
    except Exception as e:
        tag_processing_log.append(f"⚠️ Pushover notification failed: {e}")

# --- NEW: Comparator Functionality ---

def update_notion_comparator(page_id, result, output_column):
    try:
        notion.pages.update(
            page_id=page_id,
            properties={
                output_column: {"multi_select": [{"name": result}]}
            }
        )
    except Exception as e:
        comparator_processing_log.append(f"❌ Error updating Notion comparator: {e}")

def get_entries_for_comparator(notion_db_id, output_column):
    try:
        response = query_database(
            notion_db_id,
            filter={
                "and": [
                    {"property": output_column, "multi_select": {"is_empty": True}}
                ]
            }
        )
        return response.get("results", [])
    except Exception as e:
        comparator_processing_log.append(f"❌ Error fetching comparator entries: {e}")
        return []

def process_comparator_entries(notion_db_id, comparator_config):
    comparator_processing_log.append("🚀 Starting comparator process...")
    output_column = comparator_config.get("output_column_name", "").strip()
    context_columns_str = comparator_config.get("context_columns", "").strip()
    if not output_column or not context_columns_str:
        comparator_processing_log.append("❌ Error: Comparator config missing output column or context columns.")
        return
    context_columns = [col.strip() for col in context_columns_str.split(",") if col.strip()]
    entries = get_entries_for_comparator(notion_db_id, output_column)
    total = len(entries)
    comparator_processing_log.append(f"ℹ️ Found {total} entries to process.")
    count = 0
    for entry in entries:
        if stop_comparator_processing:
            comparator_processing_log.append("⏹️ Comparator process stopped by user.")
            break
        page_id = entry["id"]
        values = []
        for col in context_columns:
            try:
                prop = entry["properties"].get(col, {})
                if prop.get("type") == "rich_text":
                    text = " ".join([item.get("plain_text", "") for item in prop.get("rich_text", [])])
                    if text:
                        values.append(text)
                elif prop.get("type") == "title":
                    text = " ".join([item.get("plain_text", "") for item in prop.get("title", [])])
                    if text:
                        values.append(text)
                elif prop.get("type") == "multi_select":
                    selected = [item.get("name", "") for item in prop.get("multi_select", [])]
                    if selected:
                        values.extend(selected)
            except Exception as e:
                comparator_processing_log.append(f"❌ Error reading column {col} for entry {page_id}: {e}")
        if not values:
            comparator_processing_log.append(f"⚠️ No context values for entry {page_id}. Skipping.")
            continue
        freq = {}
        for val in values:
            freq[val] = freq.get(val, 0) + 1
        max_count = max(freq.values())
        candidates = [k for k, v in freq.items() if v == max_count]
        if len(candidates) == 1:
            result = candidates[0]
        else:
            result = "??"
        update_notion_comparator(page_id, result, output_column)
        count += 1
        comparator_processing_log.append(f"✅ Processed entry {page_id} ({count}/{total}). Result: {result}")
        time.sleep(1)
    comparator_processing_log.append("🏁 Comparator processing completed.")

def repeated_process_comparator_entries(repeat_count, notion_db_id, comparator_config):
    global stop_comparator_processing
    try:
        from app.pushover_client import send_pushover
        pushover_cfg = get_pushover_config_for_thread()
        if pushover_cfg.get('start_process'):
            send_pushover(f"Comparator process started.", title="NotyVisualScan", priority=0)
    except Exception as e:
        comparator_processing_log.append(f"⚠️ Pushover notification failed: {e}")
        pushover_cfg = {}
    
    for i in range(repeat_count):
        if stop_comparator_processing:
            comparator_processing_log.append("⏹️ Comparator process stopped by user during repetitions.")
            break
        comparator_processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        try:
            process_comparator_entries(notion_db_id, comparator_config)
            comparator_processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")
        except Exception as e:
            error_msg = f"❌ Error in repetition {i+1}: {e}"
            comparator_processing_log.append(error_msg)
            try:
                from app.pushover_client import send_pushover
                send_pushover(error_msg, title="NotyVisualScan Error", priority=1)
            except:
                pass
            continue
    
    try:
        if pushover_cfg.get('end_process'):
            from app.pushover_client import send_pushover
            send_pushover(f"Comparator process completed. Total repetitions: {repeat_count}", title="NotyVisualScan", priority=0)
    except Exception as e:
        comparator_processing_log.append(f"⚠️ Pushover notification failed: {e}")

# --- NEW: AI Comparator Functionality ---

def update_notion_ai_comparator(page_id, result, output_column, output_type):
    try:
        if output_type == "text":
            notion.pages.update(
                page_id=page_id,
                properties={
                    output_column: {"rich_text": [{"text": {"content": result}}]}
                }
            )
        else:  # Se asume "multi_select"
            notion.pages.update(
                page_id=page_id,
                properties={
                    output_column: {"multi_select": [{"name": result}]}
                }
            )
    except Exception as e:
        ai_comparator_processing_log.append(f"❌ Error updating Notion AI comparator: {e}")

def get_entries_for_ai_comparator(notion_db_id, output_column, output_type):
    try:
        if output_type == "multi_select":
            filter_query = {"property": output_column, "multi_select": {"is_empty": True}}
        elif output_type == "text":
            filter_query = {"property": output_column, "rich_text": {"is_empty": True}}
        else:
            # Si el usuario introduce un valor no reconocido, se usa el filtro de texto
            filter_query = {"property": output_column, "rich_text": {"is_empty": True}}
        response = query_database(
            notion_db_id,
            filter=filter_query
        )
        return response.get("results", [])
    except Exception as e:
        ai_comparator_processing_log.append(f"❌ Error fetching AI comparator entries: {e}")
        return []

def process_ai_comparator_entries(notion_db_id, ai_config, prompt_text, api_choice, model, token_limit):
    ai_comparator_processing_log.append("🚀 Starting AI Comparator process...")
    output_column = ai_config.get("output_column_name", "").strip()
    context_columns_str = ai_config.get("context_columns", "").strip()
    output_type = ai_config.get("output_type", "").strip()  # Se toma el valor configurado por el usuario
    if not output_column or not context_columns_str or not output_type:
        ai_comparator_processing_log.append("❌ Error: AI Comparator config missing output column, context columns or output type.")
        return
    context_columns = [col.strip() for col in context_columns_str.split(",") if col.strip()]
    entries = get_entries_for_ai_comparator(notion_db_id, output_column, output_type)
    total = len(entries)
    ai_comparator_processing_log.append(f"ℹ️ Found {total} entries to process for AI Comparator.")
    count = 0
    for entry in entries:
        if stop_ai_comparator_processing:
            ai_comparator_processing_log.append("⏹️ AI Comparator process stopped by user.")
            break
        page_id = entry["id"]
        values = []
        for col in context_columns:
            try:
                prop = entry["properties"].get(col, {})
                if prop.get("type") == "rich_text":
                    text = " ".join([item.get("plain_text", "") for item in prop.get("rich_text", [])])
                    if text:
                        values.append(text)
                elif prop.get("type") == "title":
                    text = " ".join([item.get("plain_text", "") for item in prop.get("title", [])])
                    if text:
                        values.append(text)
                elif prop.get("type") == "multi_select":
                    selected = [item.get("name", "") for item in prop.get("multi_select", [])]
                    if selected:
                        values.extend(selected)
            except Exception as e:
                ai_comparator_processing_log.append(f"❌ Error reading column {col} for entry {page_id}: {e}")
        if not values:
            ai_comparator_processing_log.append(f"⚠️ No context values for entry {page_id}. Skipping.")
            continue
        full_prompt = f"{prompt_text}\nContext: " + " ".join(values)
        try:
            result = analyze_tag_main(api_choice, " ".join(values), prompt_text, "", model, token_limit, 1)
        except Exception as e:
            ai_comparator_processing_log.append(f"❌ Error analyzing entry {page_id}: {e}")
            continue
        if not result:
            ai_comparator_processing_log.append(f"❌ Failed to determine output for entry {page_id}.")
            continue
        update_notion_ai_comparator(page_id, result, output_column, output_type)
        count += 1
        ai_comparator_processing_log.append(f"✅ Processed entry {page_id} ({count}/{total}). Result: {result}")
        time.sleep(1)
    ai_comparator_processing_log.append("🏁 AI Comparator processing completed.")

def repeated_process_ai_comparator_entries(repeat_count, notion_db_id, ai_config, prompt_text, api_choice, model, token_limit):
    global stop_ai_comparator_processing
    try:
        from app.pushover_client import send_pushover
        pushover_cfg = get_pushover_config_for_thread()
        if pushover_cfg.get('start_process'):
            send_pushover(f"AI Comparator process started.", title="NotyVisualScan", priority=0)
    except Exception as e:
        ai_comparator_processing_log.append(f"⚠️ Pushover notification failed: {e}")
        pushover_cfg = {}
    
    for i in range(repeat_count):
        if stop_ai_comparator_processing:
            ai_comparator_processing_log.append("⏹️ AI Comparator process stopped by user during repetitions.")
            break
        ai_comparator_processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        try:
            process_ai_comparator_entries(notion_db_id, ai_config, prompt_text, api_choice, model, token_limit)
            ai_comparator_processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")
        except Exception as e:
            error_msg = f"❌ Error in repetition {i+1}: {e}"
            ai_comparator_processing_log.append(error_msg)
            try:
                from app.pushover_client import send_pushover
                send_pushover(error_msg, title="NotyVisualScan Error", priority=1)
            except:
                pass
            continue
    
    try:
        if pushover_cfg.get('end_process'):
            from app.pushover_client import send_pushover
            send_pushover(f"AI Comparator process completed. Total repetitions: {repeat_count}", title="NotyVisualScan", priority=0)
    except Exception as e:
        ai_comparator_processing_log.append(f"⚠️ Pushover notification failed: {e}")

@app.route("/save_ai_comparator_config", methods=["POST"])
def save_ai_comparator_config():
    config_name = request.form.get("config_name", "").strip()
    context_columns = request.form.get("context_columns", "").strip()
    output_column_name = request.form.get("output_column_name", "").strip()
    output_type = request.form.get("output_type", "text").strip()  # "text" o "multi_select"
    max_tags = request.form.get("max_tags", "1").strip()
    try:
        max_tags = int(max_tags)
    except ValueError:
        max_tags = 1
    config = load_config()
    if not config_name:
        flash("❌ Configuration name is required for AI Comparator configs.", "error")
        return redirect(url_for("index"))
    new_config = {
        "config_name": config_name,
        "context_columns": context_columns,
        "output_column_name": output_column_name,
        "output_type": output_type,
        "max_tags": max_tags if output_type == "multi_select" else None
    }
    config.setdefault("output_configs", []).append(new_config)
    save_config(config)
    flash("📝 AI Comparator configuration saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_ai_comparator_config", methods=["POST"])
def delete_ai_comparator_config():
    index = request.form.get("dbid_index")
    config = load_config()
    try:
        index = int(index)
        sorted_configs = sorted(config.get("output_configs", []), key=lambda x: x.get("config_name", ""))
        if 0 <= index < len(sorted_configs):
            config["output_configs"].remove(sorted_configs[index])
            save_config(config)
            flash("🗑️ AI Comparator configuration deleted successfully.", "success")
        else:
            flash("❌ Invalid AI Comparator configuration index.", "error")
    except Exception as e:
        flash(f"❌ Error deleting AI Comparator configuration: {e}", "error")
    return redirect(url_for("index"))

@app.route("/apply_ai_comparator_config", methods=["POST"])
def apply_ai_comparator_config():
    index = request.form.get("output_config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_configs = sorted(config.get("output_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_configs):
            return jsonify({"status": "❌ Invalid configuration index for AI Comparator configs."})
        selected = sorted_configs[index]
        context_columns = selected.get("context_columns", "")
        output_column_name = selected.get("output_column_name", "")
        save_config(config)
        return jsonify({
            "status": "✅ AI Comparator configuration applied successfully.",
            "context_columns": context_columns,
            "output_column_name": output_column_name
        })
    except Exception as e:
        return jsonify({"status": f"❌ Error applying AI Comparator configuration: {e}"})

@app.route("/start_ai_comparator_process", methods=["POST"])
def start_ai_comparator_process():
    global ai_comparator_background_thread, stop_ai_comparator_processing, ai_comparator_processing_log
    stop_ai_comparator_processing = False
    ai_comparator_processing_log = []
    config = load_config()
    notion_db_id = request.form.get("output_notion_db_id_select", "").strip()
    if not notion_db_id or not is_valid_uuid(notion_db_id):
        msg = "❌ Error: Invalid Notion Database ID for AI Comparator."
        ai_comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    output_config_index = request.form.get("output_config_index")
    if not output_config_index or output_config_index.strip() == "":
        msg = "❌ Error: No AI Comparator configuration selected."
        ai_comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    try:
        index = int(output_config_index)
        sorted_configs = sorted(config.get("output_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_configs):
            msg = "❌ Error: Invalid AI Comparator configuration index."
            ai_comparator_processing_log.append(msg)
            return jsonify({"status": msg})
        ai_config = sorted_configs[index]
    except Exception as e:
        msg = f"❌ Error reading AI Comparator configuration: {e}"
        ai_comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    selected_model = request.form.get("output_model")
    api_choice = request.form.get("output_api_choice", "")
    if not selected_model:
        msg = "❌ Error: No model selected for AI Comparator process."
        ai_comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    model = selected_model
    if request.form.get("output_enable_max_tokens") == "on":
        token_limit_str = request.form.get("output_max_tokens")
        try:
            token_limit = int(token_limit_str) if token_limit_str.strip() != "" else 500
        except ValueError:
            token_limit = 500
    else:
        token_limit = None
    prompt_text = request.form.get("output_prompt_select")
    repeat_count_str = request.form.get("output_repeat_count", "1")
    try:
        repeat_count = int(repeat_count_str)
    except Exception:
        repeat_count = 1
    msg = f"🚀 Starting AI Comparator process with config: API: {api_choice}, Model: {model}, Max Tokens: {token_limit}, Repetitions: {repeat_count}"
    ai_comparator_processing_log.append(msg)
    config["notion_db_id"] = notion_db_id
    save_config(config)
    ai_comparator_background_thread = threading.Thread(
        target=repeated_process_ai_comparator_entries,
        args=(repeat_count, notion_db_id, ai_config, prompt_text, api_choice, model, token_limit)
    )
    ai_comparator_background_thread.start()
    return jsonify({"status": "🚀 AI Comparator process started."})

@app.route("/stop_ai_comparator_process", methods=["POST"])
def stop_ai_comparator_process():
    global stop_ai_comparator_processing
    stop_ai_comparator_processing = True
    return jsonify({"status": "⏹️ AI Comparator process stop requested."})

@app.route("/save_dbid", methods=["POST"])
def save_dbid():
    name = request.form.get("dbid_name")
    dbid = request.form.get("dbid")
    config = load_config()
    if name and dbid:
        config.setdefault("notion_db_ids", []).append({"name": name, "id": dbid})
        save_config(config)
        flash("📝 Notion DB ID saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_dbid", methods=["POST"])
def delete_dbid():
    index = request.form.get("dbid_index")
    try:
        index = int(index)
        config = load_config()
        if 0 <= index < len(config.get("notion_db_ids", [])):
            del config["notion_db_ids"][index]
            save_config(config)
            flash("🗑️ Notion DB ID deleted successfully.", "success")
    except Exception as e:
        flash(f"❌ Error deleting DB ID: {e}", "error")
    return redirect(url_for("index"))

@app.route("/save_model", methods=["POST"])
def save_model():
    new_model = request.form.get("new_model")
    new_model_api = request.form.get("new_model_api")
    is_reasoning = True if request.form.get("is_reasoning") == "on" else False
    reasoning_effort = request.form.get("reasoning_effort")
    if reasoning_effort.lower() == "none":
        reasoning_effort = None
    config = load_config()
    if new_model and new_model_api:
        new_model_obj = {
            "name": new_model,
            "model": new_model,
            "api": new_model_api,
            "is_reasoning": is_reasoning,
            "reasoning_effort": reasoning_effort if is_reasoning else None
        }
        config.setdefault("models_list", []).append(new_model_obj)
        config["models_list"] = sorted(config["models_list"], key=lambda x: x["name"].lower())
        save_config(config)
        flash("📝 Model saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_model", methods=["POST"])
def delete_model():
    index = request.form.get("model_index")
    try:
        index = int(index)
        config = load_config()
        if 0 <= index < len(config.get("models_list", [])):
            del config["models_list"][index]
            save_config(config)
            flash("🗑️ Model deleted successfully.", "success")
    except Exception as e:
        flash(f"❌ Error deleting model: {e}", "error")
    return redirect(url_for("index"))

@app.route("/save_prompt", methods=["POST"])
def save_prompt():
    new_prompt = request.form.get("new_prompt")
    prompt_name = request.form.get("prompt_name")
    prompt_category = request.form.get("prompt_category")
    config = load_config()
    if new_prompt and prompt_name and prompt_category:
        config["prompts"].append({
            "name": prompt_name,
            "prompt": new_prompt,
            "category": prompt_category,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        save_config(config)
        flash("📝 Prompt saved successfully.", "success")
    else:
        flash("❌ Prompt name, text and category are required.", "error")
    return redirect(url_for("index"))

@app.route("/delete_prompt", methods=["POST"])
def delete_prompt():
    index = request.form.get("prompt_index")
    try:
        index = int(index)
        config = load_config()
        sorted_prompts = sorted(config["prompts"], key=lambda x: x.get("name", ""))
        if index < 0 or index >= len(sorted_prompts):
            flash("❌ Invalid prompt index.", "error")
        else:
            prompt_to_delete = sorted_prompts[index]
            config["prompts"] = [p for p in config["prompts"] if p != prompt_to_delete]
            save_config(config)
            flash("🗑️ Prompt deleted successfully.", "success")
    except Exception as e:
        flash(f"❌ Error deleting prompt: {e}", "error")
    return redirect(url_for("index"))

@app.route("/update_prompt", methods=["POST"])
def update_prompt():
    """Update an existing prompt by index."""
    data = request.get_json() if request.is_json else request.form
    try:
        index = int(data.get("prompt_index", -1))
        new_name = data.get("prompt_name", "").strip()
        new_text = data.get("prompt_text", "").strip()
        new_category = data.get("prompt_category", "").strip()
        
        if not new_name or not new_text or not new_category:
            if request.is_json:
                return jsonify({"success": False, "error": "Name, text and category are required."}), 400
            flash("❌ Name, text and category are required.", "error")
            return redirect(url_for("index"))
        
        config = load_config()
        sorted_prompts = sorted(config["prompts"], key=lambda x: x.get("name", ""))
        
        if index < 0 or index >= len(sorted_prompts):
            if request.is_json:
                return jsonify({"success": False, "error": "Invalid prompt index."}), 400
            flash("❌ Invalid prompt index.", "error")
            return redirect(url_for("index"))
        
        prompt_to_update = sorted_prompts[index]
        # Find and update in original list
        for p in config["prompts"]:
            if p == prompt_to_update:
                p["name"] = new_name
                p["prompt"] = new_text
                p["category"] = new_category
                p["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                break
        
        save_config(config)
        
        if request.is_json:
            return jsonify({"success": True, "message": "Prompt updated successfully."})
        flash("✅ Prompt updated successfully.", "success")
    except Exception as e:
        if request.is_json:
            return jsonify({"success": False, "error": str(e)}), 500
        flash(f"❌ Error updating prompt: {e}", "error")
    return redirect(url_for("index"))

@app.route("/save_columns_config", methods=["POST"])
def save_columns_config():
    config_name = request.form.get("config_name", "").strip()
    image_column = request.form.get("image_column", "").strip()
    description_column = request.form.get("description_column", "").strip()
    config = load_config()
    if not config_name:
        flash("❌ Configuration name is required for description configs.", "error")
        return redirect(url_for("index"))
    new_config = {"config_name": config_name, "image_column": image_column, "description_column": description_column}
    config.setdefault("columns_configs", []).append(new_config)
    save_config(config)
    flash("📝 Description configuration saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_columns_config", methods=["POST"])
def delete_columns_config():
    index = request.form.get("config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_columns_configs = sorted(config["columns_configs"], key=lambda x: x.get("config_name", ""))
        if 0 <= index < len(sorted_columns_configs):
            config["columns_configs"].remove(sorted_columns_configs[index])
            save_config(config)
            flash("🗑️ Description configuration deleted successfully.", "success")
        else:
            flash("❌ Invalid configuration index for description configs.", "error")
    except Exception as e:
        flash(f"❌ Error deleting description configuration: {e}", "error")
    return redirect(url_for("index"))

@app.route("/apply_columns_config", methods=["POST"])
def apply_columns_config():
    index = request.form.get("columns_config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_columns_configs = sorted(config.get("columns_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_columns_configs):
            return jsonify({"status": "❌ Invalid configuration index for description configs."})
        selected = sorted_columns_configs[index]
        image_column = selected.get("image_column", "")
        description_column = selected.get("description_column", "")
        save_config(config)
        return jsonify({
            "status": "✅ Description configuration applied successfully.",
            "image_column": image_column,
            "description_column": description_column
        })
    except Exception as e:
        return jsonify({"status": f"❌ Error applying description configuration: {e}"})

@app.route("/save_tag_config", methods=["POST"])
def save_tag_config():
    config_name = request.form.get("config_name", "").strip()
    description_column = request.form.get("description_column_tag", "").strip()
    tag_column = request.form.get("tag_column", "").strip()
    allowed_tags = request.form.get("allowed_tags", "").strip()
    config = load_config()
    if not config_name:
        flash("❌ Configuration name is required for tag configs.", "error")
        return redirect(url_for("index"))
    new_config = {"config_name": config_name, "description_column": description_column, "tag_column": tag_column, "allowed_tags": allowed_tags}
    config.setdefault("tag_configs", []).append(new_config)
    save_config(config)
    flash("📝 Tag configuration saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_tag_config", methods=["POST"])
def delete_tag_config():
    index = request.form.get("config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_tag_configs = sorted(config.get("tag_configs", []), key=lambda x: x.get("config_name", ""))
        if 0 <= index < len(sorted_tag_configs):
            config["tag_configs"].remove(sorted_tag_configs[index])
            save_config(config)
            flash("🗑️ Tag configuration deleted successfully.", "success")
        else:
            flash("❌ Invalid configuration index for tag configs.", "error")
    except Exception as e:
        flash(f"❌ Error deleting tag configuration: {e}", "error")
    return redirect(url_for("index"))

@app.route("/apply_tag_config", methods=["POST"])
def apply_tag_config():
    index = request.form.get("tag_config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_tag_configs = sorted(config.get("tag_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_tag_configs):
            return jsonify({"status": "❌ Invalid configuration index for tag configs."})
        selected = sorted_tag_configs[index]
        desc_column = selected.get("description_column", "")
        tag_column = selected.get("tag_column", "")
        allowed_tags = selected.get("allowed_tags", "")
        save_config(config)
        return jsonify({
            "status": "✅ Tag configuration applied successfully.",
            "description_column": desc_column,
            "tag_column": tag_column,
            "allowed_tags": allowed_tags
        })
    except Exception as e:
        return jsonify({"status": f"❌ Error applying tag configuration: {e}"})

# NEW: Comparator config endpoints
@app.route("/save_comparator_config", methods=["POST"])
def save_comparator_config():
    config_name = request.form.get("config_name", "").strip()
    context_columns = request.form.get("context_columns", "").strip()
    output_column_name = request.form.get("output_column_name", "").strip()
    config = load_config()
    if not config_name:
        flash("❌ Configuration name is required for comparator configs.", "error")
        return redirect(url_for("index"))
    new_config = {
        "config_name": config_name,
        "context_columns": context_columns,
        "output_column_name": output_column_name
    }
    config.setdefault("comparator_configs", []).append(new_config)
    save_config(config)
    flash("📝 Comparator configuration saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_comparator_config", methods=["POST"])
def delete_comparator_config():
    index = request.form.get("config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_configs = sorted(config.get("comparator_configs", []), key=lambda x: x.get("config_name", ""))
        if 0 <= index < len(sorted_configs):
            config["comparator_configs"].remove(sorted_configs[index])
            save_config(config)
            flash("🗑️ Comparator configuration deleted successfully.", "success")
        else:
            flash("❌ Invalid comparator configuration index.", "error")
    except Exception as e:
        flash(f"❌ Error deleting comparator configuration: {e}", "error")
    return redirect(url_for("index"))

@app.route("/apply_comparator_config", methods=["POST"])
def apply_comparator_config():
    index = request.form.get("comparator_config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_configs = sorted(config.get("comparator_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_configs):
            return jsonify({"status": "❌ Invalid configuration index for comparator configs."})
        selected = sorted_configs[index]
        context_columns = selected.get("context_columns", "")
        output_column_name = selected.get("output_column_name", "")
        save_config(config)
        return jsonify({
            "status": "✅ Comparator configuration applied successfully.",
            "context_columns": context_columns,
            "output_column_name": output_column_name
        })
    except Exception as e:
        return jsonify({"status": f"❌ Error applying comparator configuration: {e}"})

@app.route("/start_process", methods=["POST"])
def start_process():
    global background_thread, stop_processing, processing_log
    stop_processing = False
    processing_log = []
    prompt_text = request.form.get("prompt_select")
    selected_model = request.form.get("desc_model_select")
    api_choice = request.form.get("api_choice", "")
    if not selected_model:
        processing_log.append("❌ Error: No model selected for description process.")
        return jsonify({"status": "❌ Error: No model selected for description process."})
    model = selected_model
    notion_db_id = request.form.get("notion_db_id_select", "").strip()
    if not notion_db_id or not is_valid_uuid(notion_db_id):
        processing_log.append("❌ Error: Invalid Notion Database ID for description process.")
        return jsonify({"status": "❌ Error: Invalid Notion Database ID."})
    columns_config_index = request.form.get("columns_config_index")
    if not columns_config_index or columns_config_index.strip() == "":
        processing_log.append("❌ Error: No columns configuration selected for description.")
        return jsonify({"status": "❌ Error: No columns configuration selected."})
    try:
        index = int(columns_config_index)
        config = load_config()
        sorted_columns_configs = sorted(config.get("columns_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_columns_configs):
            processing_log.append("❌ Error: Invalid columns configuration index.")
            return jsonify({"status": "❌ Error: Invalid columns configuration index."})
        columns_config = sorted_columns_configs[index]
        image_column = columns_config.get("image_column", "")
        description_column = columns_config.get("description_column", "")
    except Exception as e:
        processing_log.append(f"❌ Error reading columns configuration: {e}")
        return jsonify({"status": f"❌ Error reading columns configuration: {e}"})
    if not image_column or not description_column:
        processing_log.append("❌ Error: Image and Description column names are required.")
        return jsonify({"status": "❌ Error: Missing image or description column name in configuration."})
    if request.form.get("enable_max_tokens") == "on":
        token_limit_str = request.form.get("max_tokens")
        try:
            token_limit = int(token_limit_str) if token_limit_str.strip() != "" else 500
        except ValueError:
            token_limit = 500
    else:
        token_limit = None
    batch_flag = False
    repeat_count_str = request.form.get("repeat_count", "1")
    try:
        repeat_count = int(repeat_count_str)
    except Exception:
        repeat_count = 1
    msg = f"🚀 Starting description process with config: API: {api_choice}, Model: {model}, Max Tokens: {token_limit}, Repetitions: {repeat_count}"
    processing_log.append(msg)
    print(msg, flush=True)
    config["notion_db_id"] = notion_db_id
    config["columns"]["image"] = image_column
    config["columns"]["description"] = description_column
    config["model"] = model
    config["max_tokens"] = token_limit if token_limit is not None else DEFAULT_MAX_TOKENS
    save_config(config)
    flash("📝 Configuration saved successfully.", "success")
    background_thread = threading.Thread(
        target=repeated_process_entries,
        args=(repeat_count, notion_db_id, image_column, description_column, prompt_text, api_choice, model, token_limit, request.form.get("language_select"), batch_flag)
    )
    background_thread.start()
    return jsonify({"status": "🚀 Description processing started."})

@app.route("/stop_process", methods=["POST"])
def stop_process():
    global stop_processing
    stop_processing = True
    return jsonify({"status": "⏹️ Description processes stop requested."})

@app.route("/start_tag_process", methods=["POST"])
def start_tag_process():
    global tag_background_thread, stop_tag_processing, tag_processing_log
    stop_tag_processing = False
    tag_processing_log = []
    config = load_config()
    notion_db_id = request.form.get("tag_notion_db_id_select", "").strip()
    if not notion_db_id or not is_valid_uuid(notion_db_id):
        msg = "❌ Error: Invalid Notion Database ID for tagging."
        tag_processing_log.append(msg)
        print(msg, flush=True)
        return jsonify({"status": msg})
    tag_config_index = request.form.get("tag_config_index")
    if not tag_config_index or tag_config_index.strip() == "":
        msg = "❌ Error: No tag configuration selected."
        tag_processing_log.append(msg)
        print(msg, flush=True)
        return jsonify({"status": msg})
    try:
        index = int(tag_config_index)
        config = load_config()
        sorted_tag_configs = sorted(config.get("tag_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_tag_configs):
            msg = "❌ Error: Invalid tag configuration index."
            tag_processing_log.append(msg)
            print(msg, flush=True)
            return jsonify({"status": msg})
        tag_config = sorted_tag_configs[index]
        description_column = tag_config.get("description_column", "")
        tag_column = tag_config.get("tag_column", "")
        allowed_tags = tag_config.get("allowed_tags", "")
    except Exception as e:
        msg = f"❌ Error reading tag configuration: {e}"
        tag_processing_log.append(msg)
        print(msg, flush=True)
        return jsonify({"status": msg})
    if not description_column or not tag_column:
        msg = "❌ Error: Description and tag column names are required for tagging."
        tag_processing_log.append(msg)
        print(msg, flush=True)
        return jsonify({"status": msg})
    tag_prompt = request.form.get("tag_prompt_select")
    selected_model = request.form.get("tag_model")
    tag_api_choice = request.form.get("tag_api_choice", "")
    if not selected_model:
        msg = "❌ Error: No model selected for tagging process."
        tag_processing_log.append(msg)
        print(msg, flush=True)
        return jsonify({"status": msg})
    model = selected_model
    if request.form.get("tag_enable_max_tokens") == "on":
        token_limit_str = request.form.get("tag_max_tokens")
        try:
            token_limit = int(token_limit_str) if token_limit_str.strip() != "" else 500
        except ValueError:
            token_limit = 500
    else:
        token_limit = None
    if request.form.get("tag_enable_max_tags") == "on":
        max_tags_str = request.form.get("max_tags")
        try:
            max_tags = int(max_tags_str) if max_tags_str.strip() != "" else 1
        except ValueError:
            max_tags = 1
    else:
        max_tags = None
    repeat_count_str = request.form.get("repeat_count", "1")
    try:
        repeat_count = int(repeat_count_str)
    except Exception:
        repeat_count = 1
    msg = f"🚀 Starting tagging process with config: API: {tag_api_choice}, Model: {model}, Max Tokens: {token_limit}, Repetitions: {repeat_count}"
    tag_processing_log.append(msg)
    print(msg, flush=True)
    flash("📝 Tag configuration selected.", "success")
    tag_background_thread = threading.Thread(
        target=repeated_process_tag_entries,
        args=(repeat_count, notion_db_id, description_column, tag_prompt, allowed_tags, tag_api_choice, model, token_limit, tag_column, max_tags)
    )
    tag_background_thread.start()
    return jsonify({"status": "🚀 Tagging process started."})

@app.route("/stop_tag_process", methods=["POST"])
def stop_tag_process():
    global stop_tag_processing
    stop_tag_processing = True
    return jsonify({"status": "⏹️ Tagging processes stop requested."})

# NEW: Comparator process endpoints
@app.route("/start_comparator_process", methods=["POST"])
def start_comparator_process():
    global comparator_background_thread, stop_comparator_processing, comparator_processing_log
    stop_comparator_processing = False
    comparator_processing_log = []
    config = load_config()
    notion_db_id = request.form.get("comparator_notion_db_id_select", "").strip()
    if not notion_db_id or not is_valid_uuid(notion_db_id):
        msg = "❌ Error: Invalid Notion Database ID for comparator."
        comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    comparator_config_index = request.form.get("comparator_config_index")
    if not comparator_config_index or comparator_config_index.strip() == "":
        msg = "❌ Error: No comparator configuration selected."
        comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    try:
        index = int(comparator_config_index)
        sorted_comparator_configs = sorted(config.get("comparator_configs", []), key=lambda x: x.get("config_name", ""))
        if index < 0 or index >= len(sorted_comparator_configs):
            msg = "❌ Error: Invalid comparator configuration index."
            comparator_processing_log.append(msg)
            return jsonify({"status": msg})
        comparator_config = sorted_comparator_configs[index]
    except Exception as e:
        msg = f"❌ Error reading comparator configuration: {e}"
        comparator_processing_log.append(msg)
        return jsonify({"status": msg})
    repeat_count_str = request.form.get("repeat_count", "1")
    try:
        repeat_count = int(repeat_count_str)
    except Exception:
        repeat_count = 1
    msg = f"🚀 Starting comparator process with Repetitions: {repeat_count}"
    comparator_processing_log.append(msg)
    config["notion_db_id"] = notion_db_id
    save_config(config)
    comparator_background_thread = threading.Thread(
        target=repeated_process_comparator_entries,
        args=(repeat_count, notion_db_id, comparator_config)
    )
    comparator_background_thread.start()
    return jsonify({"status": "🚀 Comparator process started."})

@app.route("/stop_comparator_process", methods=["POST"])
def stop_comparator_process():
    global stop_comparator_processing
    stop_comparator_processing = True
    return jsonify({"status": "⏹️ Comparator process stop requested."})

@app.route("/reset_config", methods=["POST"], endpoint="reset_config")
def reset_config():
    default_config = {
        "notion_db_id": "",
        "columns": {"image": "", "description": ""},
        "model": "",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "prompts": [],
        "default_description": {"prompt": "", "language": ""},
        "tag_config": {
            "allowed_tags": "",
            "tag_prompt": "",
            "tag_column": "",
            "notion_db_id": "",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_tags": 1
        },
        "notion_db_ids": [],
        "models_list": get_default_models(),
        "columns_configs": [],
        "tag_configs": [],
        "comparator_configs": [],
        "output_configs": [],
        "file_upload_configs": [],
        "languages": ["Spanish", "English"]
    }
    save_config(default_config)
    flash("♻️ Configuration reset.", "success")
    return redirect(url_for("index"))

# --- AUTOMATION CONFIGS UNIFICADO EN CONFIG.JSON ---
# Elimina el uso de automation_configs.json y usa la clave 'automation_configs' en config.json

def load_automation_configs():
    config = load_config()
    return config.get("automation_configs", {})

def save_automation_configs(configs):
    config = load_config()
    config["automation_configs"] = configs
    save_config(config)

@app.route("/automation_configs", methods=["GET"])
def get_automation_configs():
    return jsonify(load_automation_configs())

@app.route("/automation_configs", methods=["POST"])
def set_automation_configs():
    configs = request.json.get("configs", {})
    save_automation_configs(configs)
    return jsonify({"status": "ok"})

# --- PATCH EXPORT/IMPORT para solo usar config.json ---
@app.route("/export_config", methods=["GET"])
def export_config():
    try:
        return send_file(CONFIG_FILE, as_attachment=True, download_name="config_backup.json")
    except Exception as e:
        flash(f"❌ Error exporting configuration: {e}", "error")
        return redirect(url_for("index"))

@app.route("/import_config", methods=["POST"])
def import_config():
    if "config_file" not in request.files:
        flash("❌ No file part", "error")
        return redirect(url_for("index"))
    file = request.files["config_file"]
    if file.filename == "":
        flash("❌ No selected file", "error")
        return redirect(url_for("index"))
    if file:
        try:
            content = file.read()
            json_data = json.loads(content)
            # Ensure data directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            # Save to persistent location
            with open(CONFIG_FILE, "w") as f:
                json.dump(json_data, f, indent=4)
            # Clean up legacy file if it exists
            if os.path.exists(LEGACY_CONFIG_FILE):
                try:
                    os.remove(LEGACY_CONFIG_FILE)
                except:
                    pass
            flash("✅ Configuration imported successfully.", "success")
        except Exception as e:
            flash(f"❌ Error importing configuration: {e}", "error")
    return redirect(url_for("index"))

@app.route("/", methods=["GET"])
def index():
    config = load_config()
    return render_template("index.html", config=config, version=VERSION)

@app.route("/save_language", methods=["POST"])
def save_language():
    new_language = request.form.get("new_language")
    config = load_config()
    if new_language:
        config.setdefault("languages", [])
        if new_language not in config["languages"]:
            config["languages"].append(new_language)
            save_config(config)
            flash("📝 Language saved successfully.", "success")
        else:
            flash("ℹ️ Language already exists.", "info")
    return redirect(url_for("index"))

@app.route("/delete_language", methods=["POST"])
def delete_language():
    language = request.form.get("language")
    config = load_config()
    if "languages" in config and language in config["languages"]:
        config["languages"].remove(language)
        save_config(config)
        flash("🗑️ Language deleted successfully.", "success")
    else:
        flash("❌ Language not found.", "error")
    return redirect(url_for("index"))

@app.route("/save_file_upload_config", methods=["POST"])
def save_file_upload_config():
    config_name = request.form.get("config_name", "").strip() or f"Config {int(time.time())}"
    context_column = request.form.get("upload_context_column", "").strip()
    upload_file_column = request.form.get("upload_file_column", "").strip()
    config = load_config()
    if not context_column or not upload_file_column:
        flash("❌ Both context and upload file column names are required.", "error")
        return redirect(url_for("index"))
    new_config = {
        "config_name": config_name,
        "context_column": context_column,
        "upload_file_column": upload_file_column
    }
    config.setdefault("file_upload_configs", []).append(new_config)
    save_config(config)
    flash("📝 File upload configuration saved successfully.", "success")
    return redirect(url_for("index"))

@app.route("/delete_file_upload_config", methods=["POST"])
def delete_file_upload_config():
    index = request.form.get("config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_configs = sorted(config.get("file_upload_configs", []), key=lambda x: x.get("config_name", ""))
        if 0 <= index < len(sorted_configs):
            config["file_upload_configs"].remove(sorted_configs[index])
            save_config(config)
            flash("🗑️ File upload configuration deleted successfully.", "success")
        else:
            flash("❌ Invalid configuration index for file upload configs.", "error")
    except Exception as e:
        flash(f"❌ Error deleting file upload configuration: {e}", "error")
    return redirect(url_for("index"))

# --- FILE UPLOAD TO NOTION ENDPOINT ---
file_upload_processing_log = []
file_upload_background_thread = None
stop_file_upload_processing = False

# Number of concurrent uploads (adjustable)
# NOTE: Notion API has a rate limit of ~3 requests/second.
# Each file upload requires 3 API calls (create, send, attach).
# Recommended: 1-2 workers to avoid rate limiting.
MAX_CONCURRENT_UPLOADS = 1

# Rate limiting configuration
NOTION_RATE_LIMIT_DELAY = 0.1  # Minimal delay between API calls - rely on 429 retry instead
MAX_RETRIES = 3  # Maximum retries for rate-limited requests
API_TIMEOUT = 30  # Timeout for API requests in seconds


def notion_api_request_with_retry(method, url, headers, log_prefix="", **kwargs):
    """
    Make a Notion API request with automatic retry on rate limiting (HTTP 429).
    Returns (response, error_message) tuple.
    """
    import time as time_module
    
    # Debug print
    print(f"[DEBUG] {log_prefix}Making {method} request to {url[:50]}...", flush=True)
    
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time_module.time()
            
            if method == "POST":
                response = requests.post(url, headers=headers, timeout=API_TIMEOUT, **kwargs)
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, timeout=API_TIMEOUT, **kwargs)
            else:
                response = requests.get(url, headers=headers, timeout=API_TIMEOUT, **kwargs)
            
            elapsed = time_module.time() - start_time
            print(f"[DEBUG] {log_prefix}Response: {response.status_code} in {elapsed:.2f}s", flush=True)
            
            if response.status_code == 429:
                # Rate limited - get retry-after header or use default
                retry_after = int(response.headers.get('Retry-After', 1))
                file_upload_processing_log.append(f"⏳ {log_prefix}Rate limited (429). Waiting {retry_after}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                print(f"[DEBUG] {log_prefix}Rate limited! Waiting {retry_after}s...", flush=True)
                time_module.sleep(retry_after)
                continue
            
            # Log slow requests
            if elapsed > 5:
                file_upload_processing_log.append(f"⚠️ {log_prefix}Slow response: {elapsed:.1f}s (status: {response.status_code})")
            
            return response, None
            
        except requests.exceptions.Timeout:
            elapsed = time_module.time() - start_time if 'start_time' in dir() else API_TIMEOUT
            print(f"[DEBUG] {log_prefix}TIMEOUT after {elapsed:.1f}s!", flush=True)
            file_upload_processing_log.append(f"⏳ {log_prefix}Request timeout after {elapsed:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time_module.sleep(2)
                continue
            return None, f"Request timeout after {API_TIMEOUT}s (all retries exhausted)"
        except requests.exceptions.ConnectionError as e:
            print(f"[DEBUG] {log_prefix}Connection error: {str(e)[:100]}", flush=True)
            file_upload_processing_log.append(f"❌ {log_prefix}Connection error: {str(e)[:100]} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time_module.sleep(2)
                continue
            return None, f"Connection error: {str(e)[:100]}"
        except Exception as e:
            print(f"[DEBUG] {log_prefix}Exception: {str(e)}", flush=True)
            return None, str(e)
    
    return None, "Rate limited: max retries exceeded"


def upload_single_file(file_info, upload_file_column, match_mode, context_map):
    """Upload a single file to Notion. Returns result dict with timing info."""
    global stop_file_upload_processing
    import time as time_module
    
    file_start_time = time_module.time()
    
    if stop_file_upload_processing:
        return {"file": file_info['filename'], "status": "skipped", "message": "Process stopped"}
    
    filename = file_info['filename']
    file_path = file_info['temp_path']
    mimetype = file_info['mimetype']
    base_name = os.path.splitext(filename)[0]
    
    # Log file size
    try:
        file_size = os.path.getsize(file_path)
        file_upload_processing_log.append(f"📤 Starting '{filename}' ({file_size/1024:.1f} KB)...")
    except:
        file_upload_processing_log.append(f"📤 Starting '{filename}'...")
    
    # Find matching page
    page_id = None
    if match_mode == "exact":
        page_id = context_map.get(base_name)
    else:
        for ctx_name, pid in context_map.items():
            if compare_context_approx(base_name, ctx_name):
                page_id = pid
                break
    
    if not page_id:
        try:
            os.remove(file_path)
        except:
            pass
        return {"file": filename, "status": "not_found", "base_name": base_name, "match_mode": match_mode}
    
    try:
        # Step 1: Create file upload object
        step1_start = time_module.time()
        headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        # Add delay to respect rate limits
        time.sleep(NOTION_RATE_LIMIT_DELAY)
        
        create_resp, error = notion_api_request_with_retry(
            "POST",
            "https://api.notion.com/v1/file_uploads",
            headers,
            log_prefix=f"[{filename}] Step1: ",
            json={}
        )
        step1_elapsed = time_module.time() - step1_start
        
        if error:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"Create upload failed ({step1_elapsed:.1f}s): {error}"}
        
        if not create_resp.ok:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"Create upload failed ({step1_elapsed:.1f}s): {create_resp.text}"}
        
        upload_obj = create_resp.json()
        upload_url = upload_obj.get("upload_url")
        file_upload_id = upload_obj.get("id")
        
        if not upload_url or not file_upload_id:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"No upload_url or id returned: {upload_obj}"}
        
        # Step 2: Upload file content
        upload_headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": "2022-06-28"
        }
        
        # Add delay to respect rate limits
        time.sleep(NOTION_RATE_LIMIT_DELAY)
        
        step2_start = time_module.time()
        with open(file_path, 'rb') as f:
            upload_files = {"file": (filename, f, mimetype)}
            upload_resp, error = notion_api_request_with_retry(
                "POST",
                upload_url,
                upload_headers,
                log_prefix=f"[{filename}] Step2: ",
                files=upload_files
            )
        step2_elapsed = time_module.time() - step2_start
        
        if error:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"Upload content failed ({step2_elapsed:.1f}s): {error}"}
        
        if not upload_resp.ok:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"Upload content failed ({step2_elapsed:.1f}s): {upload_resp.text}"}
        
        # Step 3: Attach file to Notion page
        step3_start = time_module.time()
        attach_headers = {
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        attach_data = {
            "properties": {
                upload_file_column: {
                    "type": "files",
                    "files": [
                        {
                            "type": "file_upload",
                            "file_upload": {"id": file_upload_id},
                            "name": filename
                        }
                    ]
                }
            }
        }
        
        # Add delay to respect rate limits
        time.sleep(NOTION_RATE_LIMIT_DELAY)
        
        attach_resp, error = notion_api_request_with_retry(
            "PATCH",
            f"https://api.notion.com/v1/pages/{page_id}",
            attach_headers,
            log_prefix=f"[{filename}] Step3: ",
            json=attach_data
        )
        step3_elapsed = time_module.time() - step3_start
        
        if error:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"Attach failed ({step3_elapsed:.1f}s): {error}"}
        
        if not attach_resp.ok:
            try:
                os.remove(file_path)
            except:
                pass
            return {"file": filename, "status": "error", "error": f"Attach failed ({step3_elapsed:.1f}s): {attach_resp.text}"}
        
        # Success - clean up temp file
        total_time = time_module.time() - file_start_time
        try:
            os.remove(file_path)
        except:
            pass
        
        return {"file": filename, "status": "uploaded", "base_name": base_name, "total_time": total_time}
        
    except Exception as e:
        total_time = time_module.time() - file_start_time
        try:
            os.remove(file_path)
        except:
            pass
        return {"file": filename, "status": "error", "error": f"Exception after {total_time:.1f}s: {str(e)}"}


def process_file_uploads_background(notion_db_id, context_column, upload_file_column, file_data_list, match_mode, context_map, max_workers):
    """
    Background thread function to process file uploads to Notion.
    
    IMPORTANT: Due to Notion's rate limit of ~3 requests/second, and each file
    requiring 3 API calls (create, send, attach), concurrent uploads can cause
    significant rate limiting. 
    
    With max_workers=1: ~2-5 seconds per file depending on network latency
    With max_workers=2+: Risk of rate limiting, automatic retry with backoff
    """
    global file_upload_processing_log, stop_file_upload_processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as time_module
    
    process_start_time = time_module.time()
    results = []
    total_files = len(file_data_list)
    completed_count = 0
    
    file_upload_processing_log.append(f"🚀 Starting upload with {max_workers} worker(s)...")
    file_upload_processing_log.append(f"ℹ️ Notion API rate limit: ~3 req/s. Each file = 3 API calls.")
    if max_workers > 1:
        file_upload_processing_log.append(f"⚠️ Multiple workers may cause rate limiting. Will retry automatically if needed.")
    file_upload_processing_log.append(f"")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(upload_single_file, file_info, upload_file_column, match_mode, context_map): file_info
            for file_info in file_data_list
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            if stop_file_upload_processing:
                file_upload_processing_log.append("⏹️ File upload process stopped by user.")
                # Cancel remaining futures
                for f in future_to_file:
                    f.cancel()
                break
            
            file_info = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                filename = result.get("file", "unknown")
                status = result.get("status", "unknown")
                total_time = result.get("total_time", 0)
                
                if status == "uploaded":
                    base_name = result.get("base_name", "")
                    file_upload_processing_log.append(f"✅ [{completed_count}/{total_files}] '{filename}' → '{base_name}' ({total_time:.1f}s)")
                elif status == "not_found":
                    base_name = result.get("base_name", "")
                    file_upload_processing_log.append(f"⚠️ [{completed_count}/{total_files}] '{filename}' - No match found for '{base_name}'")
                elif status == "error":
                    error = result.get("error", "Unknown error")
                    file_upload_processing_log.append(f"❌ [{completed_count}/{total_files}] '{filename}' - {error}")
                elif status == "skipped":
                    file_upload_processing_log.append(f"⏭️ [{completed_count}/{total_files}] '{filename}' - Skipped (process stopped)")
                    
            except Exception as e:
                completed_count += 1
                file_upload_processing_log.append(f"❌ [{completed_count}/{total_files}] '{file_info['filename']}' - Exception: {e}")
                results.append({"file": file_info['filename'], "status": "error", "error": str(e)})
    
    # Calculate total process time
    total_process_time = time_module.time() - process_start_time
    
    # Calculate total time for successful uploads
    total_upload_time = sum(r.get("total_time", 0) for r in results if r.get("status") == "uploaded")
    avg_time = total_upload_time / max(1, len([r for r in results if r.get("status") == "uploaded"]))
    
    # Final summary
    successful_uploads = len([r for r in results if r.get("status") == "uploaded"])
    not_found = len([r for r in results if r.get("status") == "not_found"])
    errors = len([r for r in results if r.get("status") == "error"])
    skipped = len([r for r in results if r.get("status") == "skipped"])
    
    file_upload_processing_log.append(f"")
    file_upload_processing_log.append(f"📊 Upload Summary: {successful_uploads} uploaded, {not_found} not found, {errors} errors, {skipped} skipped out of {total_files} files.")
    if successful_uploads > 0:
        file_upload_processing_log.append(f"⏱️ Avg per file: {avg_time:.1f}s | Total process time: {total_process_time:.1f}s")
    file_upload_processing_log.append(f"✅ File upload process completed.")
    
    # Send Pushover notification
    try:
        pushover_cfg = get_pushover_config_for_thread()
        if pushover_cfg.get('end_process'):
            from app.pushover_client import send_pushover
            send_pushover(f"File upload completed. {successful_uploads}/{total_files} uploaded in {total_process_time:.0f}s.", title="NotyVisualScan", priority=0)
    except Exception as e:
        file_upload_processing_log.append(f"⚠️ Pushover notification failed: {e}")


@app.route("/upload_files_to_notion", methods=["POST"])
def upload_files_to_notion():
    global file_upload_processing_log, file_upload_background_thread, stop_file_upload_processing
    file_upload_processing_log = []
    stop_file_upload_processing = False
    
    try:
        from app.pushover_client import send_pushover
        pushover_cfg = get_pushover_config_for_thread()
        if pushover_cfg.get('start_process'):
            send_pushover(f"File upload process started.", title="NotyVisualScan", priority=0)
    except Exception as e:
        file_upload_processing_log.append(f"⚠️ Pushover notification failed: {e}")
    
    notion_db_id = request.form.get("notion_db_id", "").strip()
    context_column = request.form.get("context_column", "").strip()
    upload_file_column = request.form.get("upload_file_column", "").strip()
    files = request.files.getlist("files[]")
    match_mode = request.form.get("match_mode", "exact").strip()
    
    # Get max_workers from form (default to MAX_CONCURRENT_UPLOADS)
    try:
        max_workers = int(request.form.get("max_workers", MAX_CONCURRENT_UPLOADS))
        max_workers = max(1, min(max_workers, 20))  # Clamp between 1 and 20
    except:
        max_workers = MAX_CONCURRENT_UPLOADS
    
    if not notion_db_id or not context_column or not upload_file_column:
        msg = "❌ Missing Notion DB ID, context column, or upload file column."
        file_upload_processing_log.append(msg)
        return jsonify({"status": msg}), 400
    
    if not files or len(files) == 0:
        msg = "❌ No files uploaded."
        file_upload_processing_log.append(msg)
        return jsonify({"status": msg}), 400
    
    file_upload_processing_log.append(f"🚀 Starting file upload process with {len(files)} files...")
    
    # Quick connectivity test to Notion API
    try:
        import time as time_module
        test_start = time_module.time()
        test_resp = requests.get("https://api.notion.com/v1/users/me", 
                                 headers={"Authorization": f"Bearer {NOTION_API_KEY}", "Notion-Version": "2022-06-28"},
                                 timeout=10)
        test_elapsed = time_module.time() - test_start
        file_upload_processing_log.append(f"🔗 Notion API connectivity: {test_elapsed:.2f}s (status: {test_resp.status_code})")
        if test_elapsed > 3:
            file_upload_processing_log.append(f"⚠️ Slow connection detected! This may affect upload speed.")
    except Exception as e:
        file_upload_processing_log.append(f"⚠️ Notion API connectivity test failed: {str(e)[:100]}")
    
    # Fetch Notion rows - only rows with EMPTY files column (optimization!)
    # This dramatically reduces the number of rows to fetch
    try:
        # Filter: only get rows where the upload file column is empty
        files_empty_filter = {
            "property": upload_file_column,
            "files": {
                "is_empty": True
            }
        }
        file_upload_processing_log.append(f"🔍 Filtering rows where '{upload_file_column}' is empty...")
        notion_rows = fetch_all_notion_rows(notion_db_id, filter_obj=files_empty_filter)
        file_upload_processing_log.append(f"ℹ️ Found {len(notion_rows)} rows with empty file column.")
        context_map = {}
        for row in notion_rows:
            prop = row["properties"].get(context_column, {})
            val = None
            if prop.get("type") == "title":
                val = " ".join([t.get("plain_text", "") for t in prop.get("title", [])]).strip()
            elif prop.get("type") == "rich_text":
                val = " ".join([t.get("plain_text", "") for t in prop.get("rich_text", [])]).strip()
            elif prop.get("type") == "multi_select":
                val = ",".join([t.get("name", "") for t in prop.get("multi_select", [])]).strip()
            if val:
                context_map[val] = row["id"]
    except Exception as e:
        msg = f"❌ Error fetching Notion DB rows: {e}"
        file_upload_processing_log.append(msg)
        return jsonify({"status": msg}), 500
    
    # Save files to temp directory so they can be accessed from background thread
    file_data_list = []
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    
    for file in files:
        filename = file.filename
        # Generate unique temp filename to avoid conflicts
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)
        file_data_list.append({
            'filename': filename,
            'temp_path': temp_path,
            'mimetype': file.mimetype or 'application/octet-stream'
        })
    
    file_upload_processing_log.append(f"📁 {len(file_data_list)} files saved to temp storage. Starting upload with {max_workers} concurrent workers...")
    
    # Start background thread with parallel processing
    file_upload_background_thread = threading.Thread(
        target=process_file_uploads_background,
        args=(notion_db_id, context_column, upload_file_column, file_data_list, match_mode, context_map, max_workers)
    )
    file_upload_background_thread.start()
    
    return jsonify({"status": "🚀 File upload process started."})


@app.route("/stop_file_upload", methods=["POST"])
def stop_file_upload():
    global stop_file_upload_processing
    stop_file_upload_processing = True
    return jsonify({"status": "⏹️ File upload process stop requested."})


def fetch_all_notion_rows(database_id, filter_obj=None):
    """Fetch all rows for a Notion database, handling pagination.
    
    Args:
        database_id: The Notion database ID
        filter_obj: Optional filter object to filter results (e.g., for empty files)
    """
    import time as time_module
    
    results = []
    start_cursor = None
    page_count = 0
    total_start = time_module.time()
    
    filter_desc = " (with filter)" if filter_obj else ""
    print(f"[DEBUG] fetch_all_notion_rows: Starting pagination for DB {database_id[:20]}{filter_desc}...", flush=True)
    file_upload_processing_log.append(f"📊 Fetching Notion database rows{filter_desc}...")

    while True:
        page_count += 1
        query_kwargs = {}
        if start_cursor:
            query_kwargs["start_cursor"] = start_cursor
        if filter_obj:
            query_kwargs["filter"] = filter_obj

        page_start = time_module.time()
        print(f"[DEBUG] fetch_all_notion_rows: Fetching page {page_count}...", flush=True)
        
        response = query_database(database_id, **query_kwargs)
        
        page_elapsed = time_module.time() - page_start
        page_results = response.get("results", [])
        results.extend(page_results)
        
        print(f"[DEBUG] fetch_all_notion_rows: Page {page_count} returned {len(page_results)} rows in {page_elapsed:.2f}s (total: {len(results)} rows)", flush=True)
        file_upload_processing_log.append(f"  📄 Page {page_count}: {len(page_results)} rows ({page_elapsed:.1f}s)")

        if response.get("has_more"):
            start_cursor = response.get("next_cursor")
        else:
            break

    total_elapsed = time_module.time() - total_start
    print(f"[DEBUG] fetch_all_notion_rows: COMPLETE - {len(results)} total rows in {page_count} pages, took {total_elapsed:.2f}s", flush=True)
    
    return results

def compare_context_approx(file_name, context_name):
    """
    Compara dos nombres con la lógica aproximada:
    - Antes del guion: debe ser igual.
    - Después del guion: igual ignorando ceros a la izquierda SOLO en la parte numérica.
    Ejemplo: LV001-FG00000002 == LV001-FG002
    """
    import re
    if '-' not in file_name or '-' not in context_name:
        return file_name == context_name
    pre_file, post_file = file_name.split('-', 1)
    pre_ctx, post_ctx = context_name.split('-', 1)
    if pre_file != pre_ctx:
        return False
    # Separar prefijo de letras y parte numérica
    def split_alpha_num(s):
        m = re.match(r"([A-Za-z]+)([0-9]+)", s)
        if m:
            return m.group(1), m.group(2)
        return s, ''
    alpha_file, num_file = split_alpha_num(post_file)
    alpha_ctx, num_ctx = split_alpha_num(post_ctx)
    if alpha_file != alpha_ctx:
        return False
    # Comparar partes numéricas ignorando ceros a la izquierda
    return num_file.lstrip('0') == num_ctx.lstrip('0')

# --- AUTOMATION STATE CAPTURE ---
@app.route('/api/automation/current-state', methods=['POST'])
def capture_automation_state():
    """
    Recibe el estado actual de la UI y lo almacena para su uso en la automatización.
    Mapea correctamente los índices de las configuraciones guardadas a sus objetos correspondientes.
    """
    global current_automation_state
    try:
        state = request.json
        config = load_config()
        
        # Para la configuración de descripción - mapea el índice a la configuración completa
        if 'description' in state and state['description'].get('config_index') is not None:
            try:
                desc_index = int(state['description']['config_index'])
                # Obtener la lista ordenada alfabéticamente (igual que en el frontend)
                sorted_columns_configs = sorted(config.get("columns_configs", []), key=lambda x: x.get("config_name", ""))
                if 0 <= desc_index < len(sorted_columns_configs):
                    # Obtener la configuración completa por índice
                    selected_config = sorted_columns_configs[desc_index]
                    # Reemplazar el índice por la configuración completa
                    state['description']['selected_config'] = selected_config
                    app.logger.info(f"Mapped description config index {desc_index} to config: {selected_config.get('config_name')}")
            except (ValueError, IndexError) as e:
                app.logger.error(f"Error mapping columns_config index: {e}")
        
        # Para la configuración de tagging
        if 'tagging' in state and state['tagging'].get('config_index') is not None:
            try:
                tag_index = int(state['tagging']['config_index'])
                sorted_tag_configs = sorted(config.get("tag_configs", []), key=lambda x: x.get("config_name", ""))
                if 0 <= tag_index < len(sorted_tag_configs):
                    selected_config = sorted_tag_configs[tag_index]
                    state['tagging']['selected_config'] = selected_config
                    app.logger.info(f"Mapped tagging config index {tag_index} to config: {selected_config.get('config_name')}")
            except (ValueError, IndexError) as e:
                app.logger.error(f"Error mapping tag_config index: {e}")
        
        # Para la configuración de comparator
        if 'comparator' in state and state['comparator'].get('config_index') is not None:
            try:
                comp_index = int(state['comparator']['config_index'])
                sorted_comparator_configs = sorted(config.get("comparator_configs", []), key=lambda x: x.get("config_name", ""))
                if 0 <= comp_index < len(sorted_comparator_configs):
                    selected_config = sorted_comparator_configs[comp_index]
                    state['comparator']['selected_config'] = selected_config
                    app.logger.info(f"Mapped comparator config index {comp_index} to config: {selected_config.get('config_name')}")
            except (ValueError, IndexError) as e:
                app.logger.error(f"Error mapping comparator_config index: {e}")
        
        # Para la configuración de AI comparator
        if 'ai_comparator' in state and state['ai_comparator'].get('config_index') is not None:
            try:
                ai_index = int(state['ai_comparator']['config_index'])
                sorted_output_configs = sorted(config.get("output_configs", []), key=lambda x: x.get("config_name", ""))
                if 0 <= ai_index < len(sorted_output_configs):
                    selected_config = sorted_output_configs[ai_index]
                    state['ai_comparator']['selected_config'] = selected_config
                    app.logger.info(f"Mapped AI comparator config index {ai_index} to config: {selected_config.get('config_name')}")
            except (ValueError, IndexError) as e:
                app.logger.error(f"Error mapping output_config index: {e}")
        
        # Para la configuración de File Upload
        if 'file_upload' in state and state['file_upload'].get('config_index') is not None:
            try:
                file_index = int(state['file_upload']['config_index'])
                sorted_file_configs = sorted(config.get("file_upload_configs", []), key=lambda x: x.get("config_name", ""))
                if 0 <= file_index < len(sorted_file_configs):
                    selected_config = sorted_file_configs[file_index]
                    state['file_upload']['selected_config'] = selected_config
                    app.logger.info(f"Mapped File Upload config index {file_index} to config: {selected_config.get('config_name')}")
            except (ValueError, IndexError) as e:
                app.logger.error(f"Error mapping file_upload_config index: {e}")
        
        current_automation_state = state
        return jsonify({"status": "OK", "message": "Automation state captured successfully"})
    except Exception as e:
        app.logger.error(f"Error capturing automation state: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- AUTOMATION ENDPOINTS ---
from flask import request, jsonify
import threading

automation_processing_log = []
automation_background_thread = None
stop_automation_processing = False

def run_automation_steps(steps):
    global automation_processing_log, stop_automation_processing, current_automation_state
    from time import sleep
    config = load_config()
    pushover_cfg = get_pushover_config_for_thread()
    if pushover_cfg.get('start_automation'):
        send_pushover("Automation started.", title="NotyVisualScan", priority=0)
    automation_processing_log.append("🚀 Starting automation...")
    
    # Resumen de configuraciones que se utilizarán
    if current_automation_state:
        automation_processing_log.append("📋 CONFIGURATIONS SELECTED:")
        if 'description' in current_automation_state and 'selected_config' in current_automation_state.get('description', {}):
            name = current_automation_state['description']['selected_config'].get('config_name', 'unnamed')
            automation_processing_log.append(f"  • Description: {name}")
        if 'tagging' in current_automation_state and 'selected_config' in current_automation_state.get('tagging', {}):
            name = current_automation_state['tagging']['selected_config'].get('config_name', 'unnamed')
            automation_processing_log.append(f"  • Tagging: {name}")
        if 'comparator' in current_automation_state and 'selected_config' in current_automation_state.get('comparator', {}):
            name = current_automation_state['comparator']['selected_config'].get('config_name', 'unnamed')
            automation_processing_log.append(f"  • Comparator: {name}")
        if 'ai_comparator' in current_automation_state and 'selected_config' in current_automation_state.get('ai_comparator', {}):
            name = current_automation_state['ai_comparator']['selected_config'].get('config_name', 'unnamed')
            automation_processing_log.append(f"  • AI Comparator: {name}")
        if 'file_upload' in current_automation_state and 'selected_config' in current_automation_state.get('file_upload', {}):
            name = current_automation_state['file_upload']['selected_config'].get('config_name', 'unnamed')
            automation_processing_log.append(f"  • File Upload: {name}")
        automation_processing_log.append("—————————————————————————————————")
    for idx, step in enumerate(steps):
        if stop_automation_processing:
            automation_processing_log.append("⏹️ Automation stopped by user.")
            break
        step_type = step.get('type')
        step_config = step.get('config')
        automation_processing_log.append(f"➡️ Step {idx+1}: {step_type} started.")
        try:
            if step_type == 'description':
                # Usar la configuración seleccionada desde la UI si está disponible
                if current_automation_state and 'description' in current_automation_state and 'selected_config' in current_automation_state['description']:
                    # Usar la configuración seleccionada directamente (ya resuelto el índice)
                    selected_config = current_automation_state['description']['selected_config']
                    config_name = selected_config.get('config_name', 'unnamed')
                    automation_processing_log.append(f"ℹ️ Using selected configuration: {config_name}")
                    
                    # Registros de depuración
                    automation_processing_log.append(f"🔍 Debug - Config index from UI: {current_automation_state['description'].get('config_index')}")
                    automation_processing_log.append(f"🔍 Debug - Selected config: {config_name}")
                    
                    # Obtener los valores de la configuración seleccionada
                    notion_db_id = current_automation_state['description'].get('notion_db_id') or config.get('notion_db_id', '')
                    image_column = selected_config.get('image_column', '')
                    description_column = selected_config.get('description_column', '')
                    prompt = current_automation_state['description'].get('prompt_text') or ''
                    api_choice = current_automation_state['description'].get('api_choice') or ''
                    model = current_automation_state['description'].get('model') or ''
                    token_limit = int(current_automation_state['description'].get('token_limit') or 500)
                    language = current_automation_state['description'].get('language') or ''
                    repeat_count = int(current_automation_state['description'].get('repeat_count') or 1)
                else:
                    # Fallback al comportamiento anterior si no hay estado capturado
                    columns_config = step_config
                    notion_db_id = columns_config.get('notion_db_id') or config.get('notion_db_id', '')
                    image_column = columns_config.get('image_column', '')
                    description_column = columns_config.get('description_column', '')
                    prompt = columns_config.get('prompt') or ''
                    api_choice = columns_config.get('api_choice') or ''
                    model = columns_config.get('model') or ''
                    token_limit = columns_config.get('max_tokens') or 500
                    language = columns_config.get('language') or ''
                    repeat_count = int(columns_config.get('repeat_count', 1))
                    
                batch_flag = False
                for i in range(repeat_count):
                    if stop_automation_processing:
                        automation_processing_log.append("⏹️ Automation stopped by user during description repetitions.")
                        break
                    automation_processing_log.append(f"🔄 Description repetition {i+1} of {repeat_count}...")
                    process_entries_background(notion_db_id, image_column, description_column, prompt, api_choice, model, token_limit, language, batch_flag)
                    automation_processing_log.append(f"✅ Description repetition {i+1} finished.")
            elif step_type == 'tagging':
                # Usar la configuración seleccionada desde la UI si está disponible
                if current_automation_state and 'tagging' in current_automation_state and 'selected_config' in current_automation_state['tagging']:
                    # Usar la configuración seleccionada directamente (ya resuelto el índice)
                    selected_config = current_automation_state['tagging']['selected_config']
                    config_name = selected_config.get('config_name', 'unnamed')
                    automation_processing_log.append(f"ℹ️ Using selected tagging configuration: {config_name}")
                    
                    # Registros de depuración
                    automation_processing_log.append(f"🔍 Debug - Tagging config index from UI: {current_automation_state['tagging'].get('config_index')}")
                    automation_processing_log.append(f"🔍 Debug - Selected tagging config: {config_name}")
                    
                    # Obtener valores de la configuración seleccionada
                    notion_db_id = current_automation_state['tagging'].get('notion_db_id') or config.get('notion_db_id', '')
                    description_column = selected_config.get('description_column', '')
                    tag_column = selected_config.get('tag_column', '')
                    allowed_tags = selected_config.get('allowed_tags', '')
                    tag_prompt = current_automation_state['tagging'].get('prompt_text') or ''
                    api_choice = current_automation_state['tagging'].get('api_choice') or ''
                    model = current_automation_state['tagging'].get('model') or ''
                    token_limit = int(current_automation_state['tagging'].get('token_limit') or 500)
                    max_tags = int(current_automation_state['tagging'].get('max_tags') or 1)
                    repeat_count = int(current_automation_state['tagging'].get('repeat_count') or 1)
                else:
                    # Fallback al comportamiento anterior
                    tag_config = step_config
                    notion_db_id = tag_config.get('notion_db_id') or config.get('notion_db_id', '')
                    description_column = tag_config.get('description_column', '')
                    tag_column = tag_config.get('tag_column', '')
                    allowed_tags = tag_config.get('allowed_tags', '')
                    tag_prompt = tag_config.get('tag_prompt') or ''
                    api_choice = tag_config.get('api_choice') or ''
                    model = tag_config.get('model') or ''
                    token_limit = tag_config.get('max_tokens') or 500
                    max_tags = tag_config.get('max_tags') or 1
                    repeat_count = int(tag_config.get('repeat_count', 1))
                for i in range(repeat_count):
                    if stop_automation_processing:
                        automation_processing_log.append("⏹️ Automation stopped by user during tagging repetitions.")
                        break
                    automation_processing_log.append(f"🔄 Tagging repetition {i+1} of {repeat_count}...")
                    process_tag_entries_background(notion_db_id, description_column, tag_prompt, allowed_tags, api_choice, model, token_limit, tag_column, max_tags)
                    automation_processing_log.append(f"✅ Tagging repetition {i+1} finished.")
            elif step_type == 'comparator':
                # Usar la configuración seleccionada desde la UI si está disponible
                if current_automation_state and 'comparator' in current_automation_state and 'selected_config' in current_automation_state['comparator']:
                    # Usar la configuración seleccionada directamente (ya resuelto el índice)
                    selected_config = current_automation_state['comparator']['selected_config']
                    config_name = selected_config.get('config_name', 'unnamed')
                    automation_processing_log.append(f"ℹ️ Using selected comparator configuration: {config_name}")
                    
                    # Registros de depuración
                    automation_processing_log.append(f"🔍 Debug - Comparator config index from UI: {current_automation_state['comparator'].get('config_index')}")
                    automation_processing_log.append(f"🔍 Debug - Selected comparator config: {config_name}")
                    
                    # Obtener la configuración
                    comparator_config = selected_config
                    notion_db_id = current_automation_state['comparator'].get('notion_db_id') or config.get('notion_db_id', '')
                    repeat_count = int(current_automation_state['comparator'].get('repeat_count') or 1)
                else:
                    # Fallback al comportamiento anterior
                    comparator_config = step_config
                    notion_db_id = comparator_config.get('notion_db_id') or config.get('notion_db_id', '')
                    repeat_count = int(comparator_config.get('repeat_count', 1))
                for i in range(repeat_count):
                    if stop_automation_processing:
                        automation_processing_log.append("⏹️ Automation stopped by user during comparator repetitions.")
                        break
                    automation_processing_log.append(f"🔄 Comparator repetition {i+1} of {repeat_count}...")
                    process_comparator_entries(notion_db_id, comparator_config)
                    automation_processing_log.append(f"✅ Comparator repetition {i+1} finished.")
            elif step_type == 'ai_comparator':
                # Usar la configuración seleccionada desde la UI si está disponible
                if current_automation_state and 'ai_comparator' in current_automation_state and 'selected_config' in current_automation_state['ai_comparator']:
                    # Usar la configuración seleccionada directamente (ya resuelto el índice)
                    selected_config = current_automation_state['ai_comparator']['selected_config']
                    config_name = selected_config.get('config_name', 'unnamed')
                    automation_processing_log.append(f"ℹ️ Using selected AI comparator configuration: {config_name}")
                    
                    # Registros de depuración
                    automation_processing_log.append(f"🔍 Debug - AI comparator config index from UI: {current_automation_state['ai_comparator'].get('config_index')}")
                    automation_processing_log.append(f"🔍 Debug - Selected AI comparator config: {config_name}")
                    
                    # Obtener valores de la configuración
                    ai_config = selected_config
                    notion_db_id = current_automation_state['ai_comparator'].get('notion_db_id') or config.get('notion_db_id', '')
                    prompt_text = current_automation_state['ai_comparator'].get('prompt_text') or ''
                    api_choice = current_automation_state['ai_comparator'].get('api_choice') or ''
                    model = current_automation_state['ai_comparator'].get('model') or ''
                    token_limit = int(current_automation_state['ai_comparator'].get('token_limit') or 500)
                    repeat_count = int(current_automation_state['ai_comparator'].get('repeat_count') or 1)
                else:
                    # Fallback al comportamiento anterior
                    ai_config = step_config
                    notion_db_id = ai_config.get('notion_db_id') or config.get('notion_db_id', '')
                    prompt_text = ai_config.get('prompt') or ''
                    api_choice = ai_config.get('api_choice') or ''
                    model = ai_config.get('model') or ''
                    token_limit = ai_config.get('max_tokens') or 500
                    repeat_count = int(ai_config.get('repeat_count', 1))
                for i in range(repeat_count):
                    if stop_automation_processing:
                        automation_processing_log.append("⏹️ Automation stopped by user during AI comparator repetitions.")
                        break
                    automation_processing_log.append(f"🔄 AI Comparator repetition {i+1} of {repeat_count}...")
                    process_ai_comparator_entries(notion_db_id, ai_config, prompt_text, api_choice, model, token_limit)
                    automation_processing_log.append(f"✅ AI Comparator repetition {i+1} finished.")
            elif step_type == 'file_upload':
                # Usar la configuración seleccionada desde la UI si está disponible
                if current_automation_state and 'file_upload' in current_automation_state and 'selected_config' in current_automation_state['file_upload']:
                    # Usar la configuración seleccionada directamente (ya resuelto el índice)
                    selected_config = current_automation_state['file_upload']['selected_config']
                    config_name = selected_config.get('config_name', 'unnamed')
                    automation_processing_log.append(f"ℹ️ Using selected File Upload configuration: {config_name}")
                    
                    # Registros de depuración
                    automation_processing_log.append(f"🔍 Debug - File Upload config index from UI: {current_automation_state['file_upload'].get('config_index')}")
                    automation_processing_log.append(f"🔍 Debug - Selected File Upload config: {config_name}")
                    
                    # Configuración de file upload
                    file_upload_config = selected_config
                    notion_db_id = current_automation_state['file_upload'].get('notion_db_id') or config.get('notion_db_id', '')
                    # Implementar el procesamiento real de carga de archivos aquí
                else:
                    # Fallback al comportamiento anterior
                    file_upload_config = step_config
                
                automation_processing_log.append("[File Upload process would be executed with the selected configuration]")
            else:
                automation_processing_log.append(f"❌ Unknown process type: {step_type}")
        except Exception as e:
            automation_processing_log.append(f"❌ Error in step {idx+1}: {e}")
            send_pushover(f"Automation error: {e}", title="NotyVisualScan Error", priority=1)
            break
        automation_processing_log.append(f"✅ Step {idx+1}: {step_type} finished.")
        sleep(1)
    automation_processing_log.append("🏁 Automation finished.")
    if pushover_cfg.get('end_automation'):
        send_pushover("Automation finished.", title="NotyVisualScan", priority=0)

@app.route('/start_automation', methods=['POST'])
def start_automation():
    global automation_background_thread, stop_automation_processing, automation_processing_log
    stop_automation_processing = False
    automation_processing_log = []
    steps = request.json.get('steps', [])
    # Nota: current_automation_state ya contiene el estado capturado por /api/automation/current-state
    automation_background_thread = threading.Thread(target=run_automation_steps, args=(steps,))
    automation_background_thread.start()
    return jsonify({'status': '🚀 Automation started.'})

@app.route('/stop_automation', methods=['POST'])
def stop_automation():
    global stop_automation_processing
    stop_automation_processing = True
    return jsonify({'status': '⏹️ Automation stop requested.'})

@app.route('/automation_logs', methods=['GET'])
def automation_logs():
    global automation_processing_log
    return jsonify({'log': '\n'.join(automation_processing_log)})

# --- Pushover Notification Config API ---
@app.route('/api/pushover-notif-config', methods=['GET'])
def get_pushover_notif_config():
    config = load_config()
    # Default config if not set
    default = {
        'start_process': False,
        'end_process': True,
        'start_automation': False,
        'end_automation': True
    }
    return jsonify(config.get('pushover_notif_config', default))

@app.route('/api/pushover-notif-config', methods=['POST'])
def set_pushover_notif_config():
    config = load_config()
    data = request.get_json(force=True)
    # Only allow the expected keys
    allowed = {'start_process', 'end_process', 'start_automation', 'end_automation'}
    pushover_cfg = {k: bool(data.get(k, False)) for k in allowed}
    config['pushover_notif_config'] = pushover_cfg
    save_config(config)
    return jsonify({'ok': True, 'pushover_notif_config': pushover_cfg})
