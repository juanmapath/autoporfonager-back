from django.db import models
from django.core.exceptions import ValidationError


class TimeStampedModel(models.Model):
    """Abstract base model with created_at and updated_at timestamps."""
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class ImmutableModel(models.Model):
    """
    Abstract base model for immutable audit/event records.
    Records can only be inserted, never updated or deleted.
    """
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if self.pk is not None and not kwargs.get('force_insert', False):
            # Check if this object already exists in database
            if self.__class__.objects.filter(pk=self.pk).exists():
                raise ValidationError(f"{self.__class__.__name__} is immutable and cannot be updated.")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValidationError(f"{self.__class__.__name__} is immutable and cannot be deleted.")
