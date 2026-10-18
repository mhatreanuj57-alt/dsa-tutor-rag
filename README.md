# DSA Tutor RAG — C++, Python & Java

**Live app:** [Open DSA Tutor](https://dsa-tutor-rag.streamlit.app/)

A study assistant that answers from your own DSA implementations and notes, using
vector search, BM25, reciprocal rank fusion, and cross-encoder reranking.

- Index .cpp, .py, .java, .md, and .txt learning material.
- Choose C++, Python, Java, or all languages in the Streamlit interface.
- Keep separate conversations per language and export them as JSON.
- Inspect the exact retrieved chunks with matching syntax highlighting.
- Ask from the CLI with the same language filter.

## Setup

Use Python 3.10+ and install the existing dependencies:

~~~powershell
Set-Location -LiteralPath 'C:\Users\ADMIN\Projects\dsa-tutor-rag'
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
~~~

Set GROQ_API_KEY in .env and keep GROQ_MODEL=openai/gpt-oss-120b. The client uses
Groq's fixed API endpoint and rejects other model selections before sending a
request. There is no fallback model. Keep your account on Groq's **Free plan**;
the same model incurs charges on a paid plan, and the app cannot verify your
billing tier. Free-plan quotas apply; quota errors stop without automatic retries.
See [Groq's current limits](https://console.groq.com/docs/rate-limits).
Restart Streamlit after changing .env so its cached client uses the new key.
Never commit the key. Indexing and retrieval do not require a chat API
key. The embedding and reranking models need an initial download and sufficient
local memory.

## Index the three local collections

The source folders remain independent. Ingestion reads the local collections;
deployment uses a published snapshot of their indexed passages:

~~~powershell
python ingest.py --cpp-dir '..\DSA-C--Notes' --python-dir '..\DS with python' --java-dir '..\DS with java'
~~~

You can substitute other paths or index just one language. Each build synchronizes
the **complete** active corpus: include all three directories when refreshing all
three languages. Files removed from the supplied corpus, including shortened
files' obsolete chunks, are removed from the index after successful upserts.
Source reading and embedding complete before index writes begin. A failed write
can leave a partially refreshed index; rerun the full ingestion command to finish.

For an existing single mixed corpus, these original commands remain supported:

~~~powershell
python ingest.py
python ingest.py --notes-dir '.\data\notes'
~~~

The --force option remains accepted for compatibility; all builds now perform a
full synchronization without deleting the collection first. Existing indexes
should be rebuilt once to attach language metadata.

Sources are namespaced, for example python/graphs/dijkstra.py and
java/src/graphs/Dijkstra.java, so similarly named files cannot overwrite each
other. Documentation in an explicitly selected language folder inherits its
language. Java's src/ directory is omitted from the topic label, not the citation.
Generated output, hidden directories, virtual environments, caches, and the
Python collection verification runner are excluded. Indexing never executes
source code.

## Use the tutor

~~~powershell
python -m streamlit run app.py
python tutor.py --language python 'How does binary search work?'
python tutor.py --language java 'Explain Dijkstra and its restrictions'
python tutor.py --language cpp 'How does Kruskal work?'
python tutor.py --stats
~~~

Omit --language to search the complete corpus. The language filter applies to both
vector and lexical candidates. The prompt preserves the source language and
instructs the model to cite full source paths. Retrieved-file lists identify
supplied evidence; they are not an independent verification of every generated
claim. No context returns "This isn't in my notes yet" without calling the model.

## Interface

The light study workspace uses Google's familiar blue, red, yellow, and green
palette, Material Outlined icons, suggested questions, and an inspectable source
panel. The icon font is copied from the user's local material-design-icons
collection and bundled under its Apache 2.0 license at
assets/material-icons-LICENSE.txt. No icon CDN is required. Roboto is requested
from Google Fonts; a sans-serif fallback remains available offline.

Theme settings live in .streamlit/config.toml and layout styles in assets/ui.css.
Restart Streamlit after changing theme configuration. The UI requires Streamlit
1.62 or newer. Entrance and hover effects respect reduced-motion preferences.
This is an independent study project, not a Google product.

## Streamlit hosting

Follow [the deployment guide](docs/streamlit-deploy.md). Select Python 3.13 and
store GROQ_API_KEY in Community Cloud's Secrets settings. A fresh database
restores data/index-snapshot.json.gz without rebuilding embeddings or needing
access to your Windows folders. An existing nonempty index is preserved.

The snapshot contains the indexed learning text and vectors, so it is part of
the published project. Refresh it after re-ingestion with `python export_index.py`.
See [snapshot provenance and refresh behavior](data/README.md). Requirements pin
the tested application packages and select CPU-only PyTorch on Linux. Model
downloads and memory requirements still apply to the first real question.

## Verification

~~~powershell
python -m pytest tests/test_multilingual.py
~~~

The regression suite uses a deterministic test embedder and a real temporary
Chroma database. It checks language discovery, source identities, metadata
filtering, safe rebuilds, obsolete-chunk removal, prompts, ingestion arguments,
and Streamlit conversations without model downloads or paid API calls. This
verifies integration behavior, not semantic retrieval or answer quality.

The original test_hybrid.py, test_retrieval.py, and test_e2e.py scripts remain
manual smoke checks. They require a populated index; the end-to-end script also
requires a working provider key. The src/eval.py judge is an exploratory model
evaluation, not proof of answer accuracy.
