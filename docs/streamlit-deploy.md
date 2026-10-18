# Deploy DSA Tutor on Streamlit Community Cloud

Push all ten batches before deploying. The packaged learning snapshot is part of
the repository, while `.env`, `.streamlit/secrets.toml`, and `db/` remain ignored.

Open the deployment dashboard from PowerShell:

```powershell
Start-Process 'https://share.streamlit.io/'
```

Sign in with GitHub as `mhatreanuj57-alt` and choose **Create app**.
Use these settings:

| Field | Value |
| --- | --- |
| Repository | mhatreanuj57-alt/dsa-tutor-rag |
| Branch | main |
| Main file path | app.py |
| Python version (Advanced settings) | 3.13 |

Paste the following into **Advanced settings > Secrets**, replacing only the
placeholder with your private Groq key. Rotate any key previously shared in chat.
Keep these values at the TOML root, without a section heading:

```toml
GROQ_API_KEY = "your-private-groq-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Keep your Groq account on the Free plan. The app pins one model but cannot verify
your billing tier; using the same model on a paid account can incur charges.

Save the settings and click **Deploy**. There is no Community Cloud deployment
command in the installed Streamlit CLI: `streamlit run app.py` starts a local app.

On a fresh host, the first database access restores all 2,287 packaged passages
without model downloads or re-embedding. The first question downloads and loads
the embedding and reranking models; this is slower than subsequent questions.
The Linux dependency list selects CPU-only PyTorch to avoid CUDA packages.

Check Python, Java, and C++ questions and expand a source. A missing-key error
means the root-level secrets were not saved. A 429 error means Groq's quota was
reached: wait for it to reset; there is no paid fallback. Memory errors or model
download failures need investigation in **Manage app > Logs**; local checks do
not establish Community Cloud's available memory or network access.

The app-local database is disposable. See [data/README.md](../data/README.md) for
refreshing the published corpus and the distinction between fresh and existing
databases.

Official references: [deploy](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management),
[dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
