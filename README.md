------------------------------------------------------------
NotyVisualScan
------------------------------------------------------------

NotyVisualScan is a web application that processes images stored in a Notion database, generating detailed visual descriptions using advanced AI models. The application now includes new features to streamline configuration management, process control, and tagging flexibility.

------------------------------------------------------------
Key Features
------------------------------------------------------------
• Generate Image Descriptions:
  - Utilize APIs such as OpenAI (GPT-4 and o-series reasoning models like o3-mini), Anthropic, Google Gemini, and DeepSeek to create precise and comprehensive descriptions of images.

• Automatically Tag Images:
  - Generate tags based on the generated image description.
  - Option to limit the number of tags or let the model return all relevant tags.

• Process Control:
  - Stop all ongoing processes within a specific tab (Description or Tagging) with a single confirmation pop-up.
  - If multiple repetitions are running, they are all halted at once with one confirmation.

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
Getting Started
------------------------------------------------------------
Prerequisites:
• Docker and Docker Compose installed.
• API keys for Notion, OpenAI, Anthropic, Google Gemini, and DeepSeek. These keys must be provided as environment variables.

Initial Setup:
1. Launch the Application:
   - Once the application is running, access the web interface at http://localhost:5007.

2. Configure Models and DB Credentials:
   - In the DB Credentials Tab, add your Notion database IDs.
   - In the Models Tab, select from pre-configured models or add new ones.
     (For reasoning models, such as o3-mini, the system automatically handles parameters like "reasoning_effort".)

3. Set the Default Prompt and Language:
   - In the Prompts Tab, review and manage your custom prompts used for both image description and tagging, as well as the default language for the generated text.

4. Configure Column and Tag Settings Individually:
   - In the Description Configs tab, save configurations for column names (e.g., image column and description column). Apply or delete these configurations as needed.
   - In the Tag Configs tab, save configurations for tagging (e.g., tag column name and allowed tags). These configurations can be applied or deleted when needed.
   - In the Description and Tagging tabs, select a saved configuration from a dropdown.

5. Tagging Flexibility:
   - When initiating the tagging process, you now have the option to activate a maximum number of tags.
   - If you do not enable this option, the model will return all tags it deems relevant.

6. Process Control:
   - In both the Description and Tagging tabs, you can stop all running processes within that tab with one confirmation pop-up – even if multiple repetitions are running.

7. Backup Your Configuration:
   - Use the new Backup tab to export your current configuration as a JSON file.
   - Import a previously exported configuration file to restore your settings.

------------------------------------------------------------
Running with Docker
------------------------------------------------------------
Using Docker Compose:
Build and run the application with Docker Compose by running:

    docker-compose up --build

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
Conclusion
------------------------------------------------------------
NotyVisualScan now offers enhanced process control, flexible tagging limits, and a complete backup/restore mechanism for its configuration. These updates help you manage and scale your image processing workflow more efficiently, ensuring a smooth and customizable experience.
