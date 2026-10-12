"""Upload page: choose an image -> ranked comparison against the whole database."""
from __future__ import annotations

import logging

from django.shortcuts import render

from . import ranking, services

log = logging.getLogger(__name__)


def compare_page(request):
    ctx = {"results": [], "error": None, "searched": False, "query_text": None}
    if request.method == "POST":
        upload = request.FILES.get("image")
        kind = request.POST.get("kind", "signature")
        if kind not in ("signature", "handwriting"):
            kind = "signature"
        use_qwen = request.POST.get("use_qwen") == "on"
        ctx["use_qwen"] = use_qwen
        if not upload:
            ctx["error"] = "Please choose an image."
        else:
            try:
                top_k = max(1, min(int(request.POST.get("top_k") or 10), 100))
                _, prep = ranking.prepare_upload(upload)
                results, q_text = ranking.search(prep, kind, top_k, use_qwen)
                for r in results:
                    r["percent"] = round(r["final_score"] * 100, 1)
                ctx.update(results=results, searched=True, query_text=q_text)
            except ValueError as e:
                ctx["error"] = str(e)
            except Exception as e:
                log.exception("compare page failed")
                ctx["error"] = f"Search failed: {e}"
    return render(request, "signatures/compare.html", ctx)
