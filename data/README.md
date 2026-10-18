# Deployment learning snapshot

`index-snapshot.json.gz` contains the indexed DSA learning passages, source paths,
language metadata, and all-MiniLM-L6-v2 embedding vectors. It includes excerpts
from Anuj Mhatre's C++, Python, and Java collections. It contains no chat history
or application credentials. The C++ collection's original license is preserved
in `corpus-LICENSE.txt`; the tutor's MIT license is in the repository root.

The current snapshot has 2,287 passages: C++ 1,914; Python 138; Java 235.
These are passage counts, not file counts. The sibling source folders remain
independent; the snapshot publishes their indexed text so the hosted tutor can
show and search the same evidence without access to a Windows drive.

A fresh database restores this portable snapshot at startup. An existing
nonempty database is never overwritten automatically. Updates to this file take
effect in an existing deployment only after its app-local database is removed or
reset; restarting a process alone does not replace a nonempty database.

After refreshing the full local index with `ingest.py`, run `python export_index.py`
and commit the updated snapshot. Export rejects common credential-like strings
and preserves the previous snapshot if validation fails. The scan is not a
complete secret detector; only index material intended for publication.
