from django.db import models


class Signature(models.Model):
    KINDS = [("signature", "Signature"), ("handwriting", "Handwriting")]

    person_ref = models.CharField(max_length=64, db_index=True)    # your customer / account id
    person_name = models.CharField(max_length=255, blank=True)
    kind = models.CharField(max_length=16, choices=KINDS, default="signature", db_index=True)
    image = models.ImageField(upload_to="signatures/%Y/%m/")        # stored detector crop (PNG)
    embedding = models.BinaryField(null=True, blank=True)           # float32 bytes, lets us rebuild FAISS fast
    transcript = models.CharField(max_length=255, null=True, blank=True)   # Qwen reading; NULL = not read yet
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.person_ref} / {self.kind} #{self.pk}"
