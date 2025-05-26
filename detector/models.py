from django.db import models

class News(models.Model):
    text = models.TextField()
    is_fake = models.BooleanField(default=False)

    def __str__(self):
        return self.text[:50]
