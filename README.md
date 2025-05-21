<!-- Logo -->
<p align="center">
  <img src="app/static/logo.png" alt="NotyVisualScan Logo" width="100"/>
</p>

------------------------------------------------------------
NotyVisualScan
------------------------------------------------------------

NotyVisualScan is a web application that processes images stored in a Notion database, generating detailed visual descriptions using advanced AI models. The application now includes new features to streamline configuration management, process control, tagging flexibility, and file upload automation.

------------------------------------------------------------
Key Features
------------------------------------------------------------
• Generate Image Descriptions:
  - Utilize APIs such as OpenAI (GPT-4 and o-series reasoning models like o3-mini), Anthropic, Google Gemini, and DeepSeek to create precise and comprehensive descriptions of images.

• Automatically Tag Images:
  - Generate tags based on the generated image description.
  - Option to limit the number of tags or let the model return all relevant tags.

• File Upload Automation:
  - Upload files and automatically attach them to Notion database rows by matching file names with a context column.
  - Supports <b>Exact</b> and <b>Approximate</b> matching modes for file-to-row association.
    - <b>Suggested file/context name format:</b> <code>BD001-FG001</code>
    - <b>Exact match:</b> The file name (without extension) must be exactly the same as the context value (e.g., <code>BD001-FG001</code> matches only <code>BD001-FG001</code>).
    - <b>Approximate match:</b> The part before the dash must be identical, and the part after the dash is compared ignoring leading zeros in the numeric part (e.g., <code>BD001-FG000001</code> will match <code>BD001-FG1</code> or <code>BD001-FG001</code>).

• Manage Prompts:
  - Save, edit, and delete the prompts used to generate descriptions and tags.

• Manage Database Credentials:
  - Configure and manage your Notion database IDs.

• Configure Models:
  - Select from pre-configured models or add custom models.
  - For reasoning models (such as o3-mini), special parameters like "reasoning_effort" are automatically handled based on the model configuration.

• Individual Column and Tag Configurations:
  - Save and apply specific configurations for column names (image and description) and tag settings (tag column name and allowed tags).
  - These configurations can be applied or deleted as needed.

• Import/Export Configuration Backup:
  - Easily back up your entire configuration (including prompts, models, DB credentials, column/tag settings, and languages) by downloading a JSON file.
  - Restore your settings later by importing the file.

------------------------------------------------------------
Important Notion Setup
------------------------------------------------------------
For the application to work correctly, it is imperative that:
• The "Description" column in your Notion database is set as a Rich Text property. This column is used to store the generated image description.
• The "Tag" column in your Notion database is set as a Multi-select property. This allows the application to update it with one or more tags.

Without these configurations, the application will not function as expected.

------------------------------------------------------------
Recommendations
------------------------------------------------------------
• Column names in Notion are <b>case-sensitive</b>. Always use the exact spelling, including uppercase and lowercase letters.
• <b>Do not use spaces or special characters</b> in column names to avoid possible errors.
• For file uploads, follow the suggested naming format and choose the matching mode that best fits your use case.

------------------------------------------------------------
Getting Started
------------------------------------------------------------

### Method 1: Using Docker Compose (Recommended)

1. Make sure you have Docker and Docker Compose installed.
2. Clone this repository and navigate to the project directory.
3. Set your environment variables (see below).
4. Build and run the application:

    docker-compose up --build

5. Access the web interface at: http://localhost:5007

### Method 2: Using Python (Manual)

1. Make sure you have Python 3.8+ and pip installed.
2. Clone this repository and navigate to the project directory.
3. (Optional but recommended) Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

4. Install dependencies:

    pip install -r requirements.txt

5. Set your environment variables (see below).
6. Run the application:

    export FLASK_APP=app/app.py
    flask run --host=0.0.0.0 --port=5007

7. Access the web interface at: http://localhost:5007

------------------------------------------------------------
Environment Variables
------------------------------------------------------------
Set the following environment variables when running the app:
• SECRET_KEY
• NOTION_API_KEY
• OPENAI_API_KEY
• DEEPSEEK_API_KEY
• GEMINI_API_KEY
• ANTHROPIC_API_KEY

------------------------------------------------------------
⚠️ **IMPORTANT: Configuration Backup Notice**
------------------------------------------------------------

> **WARNING!**  
> When updating or rebuilding the Docker container, **all internal application configuration will be lost** (prompts, models, credentials, columns, tags, etc.).  
> **Always make a backup of your configuration** using the export option in the web interface before updating.  
> After updating, import the configuration file again to restore your settings.

------------------------------------------------------------
Screenshots
------------------------------------------------------------

Below are some screenshots of NotyVisualScan in action:

<p align="center">
  <img src="app/static/screenshots/Screenshot 1.png" alt="Screenshot 1" width="400"/>
  <br>Screenshot 1
</p>
<p align="center">
  <img src="app/static/screenshots/Screenshot 2.png" alt="Screenshot 2" width="400"/>
  <br>Screenshot 2
</p>
<p align="center">
  <img src="app/static/screenshots/Screenshot 3.png" alt="Screenshot 3" width="400"/>
  <br>Screenshot 3
</p>

------------------------------------------------------------
Conclusion
------------------------------------------------------------
NotyVisualScan now offers enhanced process control, flexible tagging limits, file upload automation with smart matching, and a complete backup/restore mechanism for its configuration. These updates help you manage and scale your image processing workflow more efficiently, ensuring a smooth and customizable experience.
