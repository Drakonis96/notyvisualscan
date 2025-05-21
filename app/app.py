import os
import json
import requests
import time
import threading
import uuid
import base64
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from notion_client import Client
from openai import OpenAI  # Using the new OpenAI API
import anthropic  # pip install anthropic

# Application version updated to 1.6.0
VERSION = "1.6.0"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")
app.jinja_env.globals.update(enumerate=enumerate)

# Configuration file (stored in the persisted directory)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

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

def load_config():
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
            "models_list": [
                {"name": "GPT-4o", "model": "gpt-4o", "api": "OpenAI"},
                {"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api": "OpenAI"},
                {"name": "DeepSeek Chat", "model": "deepseek-chat", "api": "DeepSeek"},
                {"name": "DeepSeek Reasoner", "model": "deepseek-reasoner", "api": "DeepSeek"},
                {"name": "Gemini 2.0 Flash", "model": "gemini-2.0-flash", "api": "Gemini"},
                {"name": "Claude 3.5 Sonnet Latest", "model": "claude-3-5-sonnet-latest", "api": "Anthropic"},
                {"name": "Claude 3.5 Haiku Latest", "model": "claude-3-5-haiku-latest", "api": "Anthropic"},
                {"name": "Claude 3 Opus Latest", "model": "claude-3-opus-latest", "api": "Anthropic"},
                {"name": "o3 Mini", "model": "o3-mini", "api": "OpenAI"},
                {"name": "o1 Mini", "model": "o1-mini", "api": "OpenAI"},
                {"name": "O1", "model": "o1", "api": "OpenAI"}
            ],
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
            config.setdefault("models_list", [])
            modelo_api_map = {
                "GPT-4o": "OpenAI",
                "GPT-4o-mini": "OpenAI",
                "DeepSeek Chat": "DeepSeek",
                "DeepSeek Reasoner": "DeepSeek",
                "Gemini 2.0 Flash": "Gemini",
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
            raise ValueError(f"❌ Gemini API error: {e}")

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
        tag_processing_log.append(f"❌ Error updating Notion tag: {e}")

def analyze_image_choice(api_choice, image_url, prompt, model, token_limit=500):
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

def analyze_tag_choice(api_choice, description_text, prompt, allowed_tags, model, max_tokens, max_tags):
    prompt = prompt.format(max_tags=max_tags)
    full_prompt = f"{prompt}\nAllowed tags: {allowed_tags}\nDescription:\n{description_text}"
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

def get_database_entries(notion_db_id, image_column, description_column):
    try:
        response = notion.databases.query(
            database_id=notion_db_id,
            filter={
                "and": [
                    {"property": description_column, "rich_text": {"is_empty": True}},
                    {"property": image_column, "files": {"is_not_empty": True}}
                ]
            }
        )
        return response.get("results", [])
    except Exception as e:
        processing_log.append(f"❌ Error fetching Notion entries: {e}")
        return []

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
        processing_log.append(f"❌ Error extracting image URL: {e}")
        return None

def get_image_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return url
    except requests.exceptions.RequestException as e:
        processing_log.append(f"❌ Error accessing image: {e}")
        return None

def analyze_image_main(api_choice, image_url, prompt, model, token_limit):
    effective_prompt = prompt.strip()  # No fallback to default prompt
    try:
        return analyze_image_choice(api_choice, image_url, effective_prompt, model, token_limit)
    except Exception as e:
        processing_log.append(str(e))
        return None

def analyze_tag_main(api_choice, description_text, prompt, allowed_tags, model, max_tokens, max_tags):
    try:
        return analyze_tag_choice(api_choice, description_text, prompt, allowed_tags, model, max_tokens, max_tags)
    except Exception as e:
        tag_processing_log.append(str(e))
        return None

def update_notion_description(page_id, description, description_column):
    try:
        notion.pages.update(
            page_id=page_id,
            properties={
                description_column: {"rich_text": [{"text": {"content": description}}]}
            }
        )
    except Exception as e:
        processing_log.append(f"❌ Error updating Notion: {e}")

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
        response = notion.databases.query(
            database_id=notion_db_id,
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
    for i in range(repeat_count):
        processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        process_entries_background(notion_db_id, image_column, description_column, prompt, api_choice, model, token_limit, language, batch_flag)
        processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")

def repeated_process_tag_entries(repeat_count, notion_db_id, description_column, tag_prompt, allowed_tags, api_choice, model, max_tokens, tag_column, max_tags):
    for i in range(repeat_count):
        tag_processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        process_tag_entries_background(notion_db_id, description_column, tag_prompt, allowed_tags, api_choice, model, max_tokens, tag_column, max_tags)
        tag_processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")

# NEW: Comparator functionality

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
        response = notion.databases.query(
            database_id=notion_db_id,
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
    for i in range(repeat_count):
        comparator_processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        process_comparator_entries(notion_db_id, comparator_config)
        comparator_processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")

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
        response = notion.databases.query(
            database_id=notion_db_id,
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
    for i in range(repeat_count):
        ai_comparator_processing_log.append(f"🔄 Repetition {i+1} of {repeat_count} starting...")
        process_ai_comparator_entries(notion_db_id, ai_config, prompt_text, api_choice, model, token_limit)
        ai_comparator_processing_log.append(f"✅ Repetition {i+1} of {repeat_count} completed.")

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
    index = request.form.get("config_index")
    config = load_config()
    try:
        index = int(index)
        sorted_configs = sorted(config.get("output_configs", []), key=lambda x: x.get("config_name", ""))
        if 0 <= index < len(sorted_configs):
            config["output_configs"].remove(sorted_configs[index])
            save_config(config)
            flash("🗑️ AI Comparator configuration deleted successfully.", "success")
        else:
            flash("❌ Invalid configuration index for AI Comparator configs.", "error")
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
        "models_list": [
            {"name": "GPT-4o", "model": "gpt-4o", "api": "OpenAI"},
            {"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api": "OpenAI"},
            {"name": "DeepSeek Chat", "model": "deepseek-chat", "api": "DeepSeek"},
            {"name": "DeepSeek Reasoner", "model": "deepseek-reasoner", "api": "DeepSeek"},
            {"name": "Gemini 2.0 Flash", "model": "gemini-2.0-flash", "api": "Gemini"},
            {"name": "Claude 3.5 Sonnet Latest", "model": "claude-3-5-sonnet-latest", "api": "Anthropic"},
            {"name": "Claude 3.5 Haiku Latest", "model": "claude-3-5-haiku-latest", "api": "Anthropic"},
            {"name": "Claude 3 Opus Latest", "model": "claude-3-opus-latest", "api": "Anthropic"},
            {"name": "o3 Mini", "model": "o3-mini", "api": "OpenAI", "is_reasoning": True, "reasoning_effort": "medium"},
            {"name": "o1 Mini", "model": "o1-mini", "api": "OpenAI", "is_reasoning": True, "reasoning_effort": None},
            {"name": "O1", "model": "o1", "api": "OpenAI", "is_reasoning": True, "reasoning_effort": None}
        ],
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
            with open(CONFIG_FILE, "w") as f:
                json.dump(json_data, f, indent=4)
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

@app.route("/upload_files_to_notion", methods=["POST"])
def upload_files_to_notion():
    global file_upload_processing_log
    file_upload_processing_log = []
    notion_db_id = request.form.get("notion_db_id", "").strip()
    context_column = request.form.get("context_column", "").strip()
    upload_file_column = request.form.get("upload_file_column", "").strip()
    files = request.files.getlist("files[]")
    # Nueva opción: coincidencia exacta o aproximada
    match_mode = request.form.get("match_mode", "exact").strip()  # "exact" o "approx"
    if not notion_db_id or not context_column or not upload_file_column:
        msg = "❌ Missing Notion DB ID, context column, or upload file column."
        file_upload_processing_log.append(msg)
        return jsonify({"status": msg}), 400
    if not files or len(files) == 0:
        msg = "❌ No files uploaded."
        file_upload_processing_log.append(msg)
        return jsonify({"status": msg}), 400
    try:
        notion_rows = notion.databases.query(database_id=notion_db_id).get("results", [])
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
        return jsonify({"status": msg, "log": file_upload_processing_log}), 500
    results = []
    for file in files:
        filename = file.filename
        base_name = os.path.splitext(filename)[0]
        page_id = None
        if match_mode == "exact":
            page_id = context_map.get(base_name)
        else:
            # Búsqueda aproximada
            for ctx_name, pid in context_map.items():
                if compare_context_approx(base_name, ctx_name):
                    page_id = pid
                    break
        if not page_id:
            msg = f"⚠️ No Notion row found for file '{filename}' (context: '{base_name}'). Skipping. (match_mode: {match_mode})"
            file_upload_processing_log.append(msg)
            results.append({"file": filename, "status": "not_found"})
            continue
        try:
            # Step 1: Create file upload object
            headers = {
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json"
            }
            create_resp = requests.post(
                "https://api.notion.com/v1/file_uploads",
                headers=headers,
                json={"filename": filename}
            )
            if not create_resp.ok:
                msg = f"❌ Error creating file upload object for '{filename}': {create_resp.text}"
                file_upload_processing_log.append(msg)

                results.append({"file": filename, "status": "error", "error": create_resp.text})
                continue
            upload_obj = create_resp.json()
            upload_url = upload_obj.get("upload_url")
            file_upload_id = upload_obj.get("id")
            if not upload_url or not file_upload_id:
                msg = f"❌ Notion did not return upload_url or id for '{filename}'. Response: {upload_obj}"
                file_upload_processing_log.append(msg)
                results.append({"file": filename, "status": "error", "error": str(upload_obj)})
                continue
            # Step 2: Upload file content
            upload_headers = {
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Notion-Version": "2022-06-28"
            }
            upload_files = {"file": (filename, file.stream, file.mimetype)}
            upload_resp = requests.post(upload_url, headers=upload_headers, files=upload_files)
            if not upload_resp.ok:
                msg = f"❌ Error uploading file content for '{filename}': {upload_resp.text}"
                file_upload_processing_log.append(msg)
                results.append({"file": filename, "status": "error", "error": upload_resp.text})
                continue
            # Step 3: Attach file to Notion page property
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
            attach_resp = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=attach_headers, json=attach_data)
            if not attach_resp.ok:
                msg = f"❌ Error attaching file to Notion for '{filename}': {attach_resp.text}"
                file_upload_processing_log.append(msg)
                results.append({"file": filename, "status": "error", "error": attach_resp.text})
                continue
            msg = f"✅ Uploaded and attached '{filename}' to Notion row '{base_name}'."
            file_upload_processing_log.append(msg)
            results.append({"file": filename, "status": "uploaded"})
        except Exception as e:
            msg = f"❌ Error uploading '{filename}': {e}"
            file_upload_processing_log.append(msg)
            results.append({"file": filename, "status": "error", "error": str(e)})
    return jsonify({"results": results, "log": file_upload_processing_log})

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
