# Agentic-AI Documentation

## Overview

This directory contains comprehensive documentation for the Agentic-AI framework. The documentation is written in Markdown and can be served using MkDocs.

## Structure

- `index.md`: Main documentation landing page
- `getting-started.md`: Quick start guide
- `architecture.md`: System architecture overview
- `tools/`: Documentation for the tool integration system
- `prompts/`: Documentation for the prompt management system
- `providers/`: Documentation for AI providers
- `conversations/`: Documentation for conversation management
- `mkdocs.yml`: MkDocs configuration file

## Building the Documentation

To build and serve the documentation locally:

1. Install MkDocs and the Material theme:

```bash
pip install mkdocs mkdocs-material
```

2. Navigate to the repository root and serve the docs:

```bash
mkdocs serve
```

3. Open your browser to http://127.0.0.1:8000

## Building Static Site

To build a static site for hosting:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

## HTML documentation

here it is: [Documentation site](./../site/index.html)
