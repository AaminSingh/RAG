# RAG Chatbot

![RAG Chatbot Banner](https://via.placeholder.com/1200x300.png?text=RAG+Chatbot)

## Overview

**RAG Chatbot** is an AI-powered conversational agent that utilizes Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant responses. Built with [LangChain](https://python.langchain.com/), [Streamlit](https://streamlit.io/), and the [Gemini API](https://gemini.com/), this chatbot offers a seamless and interactive user experience.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.0.1-green)
![Gemini API](https://img.shields.io/badge/Gemini%20API-v1-orange)

## Features

- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with generative models to enhance response accuracy.
- **Interactive UI:** User-friendly interface powered by Streamlit.
- **Modular Architecture:** Clean separation of concerns with dedicated modules for configuration, utilities, and the main application.

## File Structure

```plaintext
RAG_Chatbot/
├── app.py
├── chain.py
├── config.py
├── utils.py
├── requirements.txt
└── README.md
```

- **app.py:** Main application script; sets up the Streamlit interface and manages user interactions.
- **chain.py:** Defines the RAG pipeline using LangChain and integrates with the Gemini API.
- **config.py:** Handles configuration settings, including API keys and environment variables.
- **utils.py:** Contains helper functions for data processing and error handling.
- **requirements.txt:** Lists all Python dependencies required for the project.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/RAG_Chatbot.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd RAG_Chatbot
   ```

3. **Create a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

4. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables:**

   Create a `.env` file in the project root and add your Gemini API key:

   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Interact with the Chatbot:**

   - Open the provided local URL in your browser.
   - Enter your queries in the input field and receive responses powered by the RAG model.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes. For major updates, open an issue first to discuss your proposed changes.

## License

This project is licensed under the [MIT License](LICENSE).

---

*For any questions or support, please open an issue in this repository.*

--- 
