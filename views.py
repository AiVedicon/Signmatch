"""JSON API: enroll, search (1:N) and verify (1:1).

NOTE: endpoints are csrf_exempt for API use - put them behind authentication
(DRF token / session login / API gateway) before production.
"""
from __future__ import annotations

import logging

import cv2
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import ranking, services
from .models import Signature
from .preprocess import prepare, read_image

log = logging.getLogger(__name__)


def _error(msg: str, status: int = 400):
    return JsonResponse({"error": msg}, status=status)


def _kind(request) -> str:
    kind = request.POST.get("kind", "signature")
    if kind not in ("signature", "handwriting"):
        raise ValueError("kind must be 'signature' or 'handwriting'")
    return kind


def _use_qwen(request) -> bool:
    return request.POST.get("use_qwen") == "1" or request.POST.get("use_vlm") == "1"


@csrf_exempt
@require_POST
def enroll(request):
    """POST multipart: image, person_ref, [person_name], [kind]."""
    upload, ref = request.FILES.get("image"), request.POST.get("person_ref")
    if not upload or not ref:
        return _error("image and person_ref are required")
    try:
        kind = _kind(request)
        crop, prep = ranking.prepare_upload(upload)
    except ValueError as e:
        return _error(str(e))

    emb = services.embedder(kind).embed([prep.canvas])[0]
    ok, png = cv2.imencode(".png", crop)
    row = Signature(person_ref=ref, person_name=request.POST.get("person_name", ""),
                    kind=kind, embedding=emb.tobytes())
    row.image.save(f"{ref}.png", ContentFile(png.tobytes()), save=True)

    idx = services.index(kind)
    idx.add([row.pk], emb[None])
    idx.save()
    return JsonResponse({"id": row.pk, "person_ref": ref, "kind": kind})


@csrf_exempt
@require_POST
def search(request):
    """POST multipart: image, [kind], [top_k], [use_qwen=1].  1:N search, ranked."""
    upload = request.FILES.get("image")
    if not upload:
        return _error("image is required")
    try:
        kind = _kind(request)
        top_k = int(request.POST.get("top_k", services.cfg()["TOP_K"]))
        _, prep = ranking.prepare_upload(upload)
    except ValueError as e:
        return _error(str(e))
    try:
        results, q_text = ranking.search(prep, kind, top_k, _use_qwen(request))
    except Exception as e:                      # e.g. Qwen not installed / out of GPU memory
        log.exception("search failed")
        return _error(f"Search failed: {e}", 500)
    return JsonResponse({"kind": kind, "query_text": q_text,
                         "results": [ranking.to_json(r) for r in results]})


@csrf_exempt
@require_POST
def verify(request):
    """POST multipart: image, signature_id, [use_qwen=1].  1:1 detailed comparison."""
    upload, sid = request.FILES.get("image"), request.POST.get("signature_id")
    if not upload or not sid:
        return _error("image and signature_id are required")
    try:
        row = Signature.objects.get(pk=int(sid))
    except (ValueError, Signature.DoesNotExist):
        return _error("signature not found", 404)
    try:
        _, prep = ranking.prepare_upload(upload)
    except ValueError as e:
        return _error(str(e))
    try:
        q_emb = services.embedder(row.kind).embed([prep.canvas])[0]
        q_text = services.vlm().transcribe(prep.mask).strip() if _use_qwen(request) else None
        res = ranking.compare_with_row(prep, row, q_emb, q_text, services.cfg())
    except Exception as e:
        log.exception("verify failed")
        return _error(f"Verify failed: {e}", 500)
    return JsonResponse(ranking.to_json(res))
